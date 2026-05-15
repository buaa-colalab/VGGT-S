import argparse
import copy
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from vggt.models.vggt import VGGT

from dataloader import Egoexo4dDataTrainDataLoader, Egoexo4dDataValDataLoader
from model.predictor import Predictor
from utils import (cal_iou, dice_loss, focal_loss, get_newest_ckpt,
                   is_main_process, mask_sampling, my_show_masks_group,
                   recursive_convert, sample_feat_at_points, save_on_master,
                   setup_for_distributed)

CURRENT_DIR = Path(__file__).resolve().parent


def vggt_track(
        vggt,
        src_img_b,  # np.ndarray, shape: [B, H, W, C], uint8
        tar_img_b,  # np.ndarray, shape: [B, H, W, C], uint8
        src_points_b,  # torch.Tensor or np.ndarray, shape: [B, N, 2]
        device):
    """
    Extract VGGT tracking features from a pair of source/target images.

    Args:
        vggt:
            VGGT tracking model containing:
            - aggregator
            - track_head

        src_img_b (np.ndarray):
            Source image batch in uint8 RGB format.
            Shape: [B, H, W, C].

        tar_img_b (np.ndarray):
            Target image batch in uint8 RGB format.
            Shape: [B, H, W, C].

        src_points_b (torch.Tensor or np.ndarray):
            Query points defined on source images.
            Shape: [B, N, 2],
            where each point is (x, y).

        device:
            Torch device.

    Returns:
        feat_map_b (torch.Tensor):
            Extracted tracking feature maps from VGGT.
    """

    # Convert images to normalized torch tensors
    src_img_tb = torch.from_numpy(src_img_b / 255.0).permute(0, 3, 1,
                                                             2).float()
    tar_img_tb = torch.from_numpy(tar_img_b / 255.0).permute(0, 3, 1,
                                                             2).float()

    # Construct image pair tensor
    # Shape: [B, 2, C, H, W]
    img_pair_b = torch.stack([src_img_tb, tar_img_tb],
                             dim=1).to(device).contiguous()

    # Extract aggregated tokens
    aggregated_tokens_list, patch_start_idx = vggt.aggregator(img_pair_b)

    # Extract tracking features
    _, _, _, feat_map_b = vggt.track_head(aggregated_tokens_list,
                                          images=img_pair_b,
                                          patch_start_idx=patch_start_idx,
                                          query_points=src_points_b,
                                          feat=True)

    return feat_map_b


def vggt_feat(vggt, src_img_b, src_mask_b, tar_img_b, vggt_sample_num, device):
    """
    Sample query points from source masks and extract VGGT tracking features
    between source and target image pairs.

    Args:
        vggt:
            VGGT tracking model containing:
            - aggregator
            - track_head

        src_img_b (np.ndarray):
            Source image batch in uint8 RGB format.
            Shape: [B, H, W, C].

        src_mask_b (np.ndarray):
            Source binary masks used for point sampling.
            Shape: [B, H, W].

        tar_img_b (np.ndarray):
            Target image batch in uint8 RGB format.
            Shape: [B, H, W, C].

        vggt_sample_num (int):
            Number of sampled query points per object mask.

        device:
            Torch device.

    Returns:
        src_points_b (torch.Tensor):
            Sampled source query points.
            Shape: [B, N, 2],
            where each point is (x, y).

        feat_map_b (torch.Tensor):
            Extracted VGGT tracking feature maps.
            Example shape: [B, 2, 128, 259, 259].
    """

    # Sample query points from source masks
    src_points_b = mask_sampling(
        masks_b=src_mask_b[:, None, ...],  # [B, 1, H, W]
        n_samples=vggt_sample_num)  # [B, O, N, 2]

    # Convert to torch tensor
    src_points_b = torch.as_tensor(src_points_b).float().squeeze(1).to(
        device)  # [B, N, 2]

    # Extract VGGT tracking features
    feat_map_b = vggt_track(vggt=vggt,
                            src_img_b=src_img_b,
                            tar_img_b=tar_img_b,
                            src_points_b=src_points_b,
                            device=device)  # [B, 2, 128, 259, 259]

    feat_map_b = feat_map_b.float()

    return src_points_b, feat_map_b


train_glb_batch_idx = 0


def train_epoch(epoch, config, device, train_log_path, train_ckpt_dir, vggt,
                predictor, optimizer, lr_scheduler, train_loader, writer):
    """
    Train the predictor model for one epoch.

    Pipeline:
        1. Sample source mask points.
        2. Extract VGGT tracking features from source-target image pairs.
        3. Predict target masks with iterative refinement.
        4. Compute focal + dice segmentation losses.
        5. Backpropagate and update model parameters.
        6. Save visualizations, tensorboard logs, and checkpoints.

    Args:
        epoch (int):
            Current training epoch.

        config (dict):
            Experiment configuration dictionary.

        device:
            Torch device.

        train_log_path (str):
            Path to training log file.

        train_ckpt_dir (str):
            Directory for saving checkpoints.

        vggt:
            Frozen VGGT feature extractor.

        predictor (torch.nn.Module):
            Trainable segmentation predictor model.

        optimizer:
            Optimizer for predictor.

        lr_scheduler:
            Learning rate scheduler.

        train_loader:
            Distributed training dataloader.

        writer:
            TensorBoard SummaryWriter.
    """

    global train_glb_batch_idx

    predictor.train()

    # Number of iterations per epoch
    tot_iter = (config['exp']['train_iter'] //
                (config['data']['train_batch_size'] *
                 torch.distributed.get_world_size()))

    train_iter = iter(train_loader)

    optimizer.zero_grad()

    # ======================== training loop ========================
    for batch_idx, batch in tqdm(enumerate(train_iter),
                                 total=tot_iter,
                                 position=0,
                                 disable=not is_main_process(),
                                 dynamic_ncols=True):
        if batch_idx >= tot_iter:
            break

        (src_img_b, src_mask_b, src_img_ori_b, src_mask_ori_b, tar_img_b,
         tar_mask_b, tar_img_ori_b, tar_mask_ori_b, tar_proj_points_b,
         tar_points_b, mask_cropped, meta) = batch

        # ======================== forward ========================
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):

            # Extract frozen VGGT features
            with torch.no_grad():
                src_points_b, feat_map_b = vggt_feat(
                    vggt=vggt,
                    src_img_b=src_img_b,
                    src_mask_b=src_mask_b,
                    tar_img_b=tar_img_b,
                    vggt_sample_num=config['vggt']['sample_num'],
                    device=device)

            feat_map_b = feat_map_b.float()
            src_points_b = src_points_b.float()

            # Sample source point features
            src_points_feat_b = sample_feat_at_points(
                feat_map_b[:, 0, ...], src_points_b,
                config['data']['img_size'], config['data']['img_size'])
            src_points_feat_b = src_points_feat_b.float()

            # Ground-truth target points
            tar_points_b = torch.as_tensor(tar_points_b).float().squeeze(1).to(
                device)  # [B, N, 2]

            B = len(src_img_b)

            # Split batch into:
            #   - direct prediction samples
            #   - iterative refinement samples
            refine_num = int(B * config['exp']['refine_ratio'])

            # ==================== no refinement ==================
            pred_mask_b_0, _ = predictor(
                feat_map_b=feat_map_b[:refine_num],
                src_points_b=src_points_b[:refine_num],
                tar_points_b=tar_points_b[:refine_num],
                src_mask_b=src_mask_b[:refine_num],
                src_points_feat_b=src_points_feat_b[:refine_num],
                tar_mask_b=None,
                ret_logits=True,
            )

            pred_mask_b_0 = pred_mask_b_0.float()[:, 0, ...]  # [B, H, W]

            # ========== iterative refinement ==========
            with torch.no_grad():
                low_res_mask_b_1 = None

                for _ in range(config['exp']['refine_time']):
                    _, low_res_mask_b_1 = predictor(
                        feat_map_b=feat_map_b[refine_num:],
                        src_points_b=src_points_b[refine_num:],
                        tar_points_b=tar_points_b[refine_num:],
                        src_mask_b=src_mask_b[refine_num:],
                        src_points_feat_b=src_points_feat_b[refine_num:],
                        tar_mask_b=low_res_mask_b_1,
                        ret_logits=True,
                    )

                    low_res_mask_b_1 = low_res_mask_b_1.float()

            # Final refined prediction
            pred_mask_b_1, _ = predictor(
                feat_map_b=feat_map_b[refine_num:],
                src_points_b=src_points_b[refine_num:],
                tar_points_b=tar_points_b[refine_num:],
                src_mask_b=src_mask_b[refine_num:],
                src_points_feat_b=src_points_feat_b[refine_num:],
                tar_mask_b=low_res_mask_b_1,
                ret_logits=True,
            )

            pred_mask_b_1 = pred_mask_b_1.float()[:, 0, ...]

        # Merge predictions
        pred_mask_b = torch.cat([pred_mask_b_0, pred_mask_b_1], dim=0).float()

        # ======================== loss ========================
        gt_b = torch.as_tensor(tar_mask_b, device=device, dtype=torch.float32)

        loss_focal = focal_loss(pred_mask_b, gt_b) * 20
        loss_dice = dice_loss(pred_mask_b, gt_b)

        loss = loss_focal + loss_dice

        # ======================== backward ========================
        loss.backward()

        grad_total_norm = torch.nn.utils.clip_grad_norm_(
            predictor.parameters(), config['exp']['max_norm'])

        optimizer.step()
        optimizer.zero_grad()

        # ======================== visualization ========================
        if config['debug']['vis_batch'] - batch_idx > 0 and is_main_process():

            save_dir = (f"{CURRENT_DIR}/{config['exp']['exp_name']}"
                        f"/vis/train/epoch_{epoch}")

            for b in range(B):

                os.makedirs(save_dir, exist_ok=True)

                tar_points = tar_points_b[b].detach().cpu().numpy()
                src_points = src_points_b[b].detach().cpu().numpy()

                pred_mask = pred_mask_b[b] > 0
                pred_mask = pred_mask.detach().cpu().numpy()[None, :].astype(
                    np.uint8)

                my_show_masks_group(
                    imgs=[src_img_b[b], tar_img_b[b], tar_img_b[b]],
                    maskss=[
                        src_mask_b[b][None, :], tar_mask_b[b][None, :],
                        pred_mask
                    ],
                    pointss=[
                        src_points[None, :], tar_points[None, :],
                        tar_points[None, :]
                    ],
                    save_dir=save_dir,
                    save_name=f'{batch_idx}_{b}.jpg')

        # ======================== tensorboard ========================
        if writer is not None:
            writer.add_scalar('Loss', loss.item(), train_glb_batch_idx)
            writer.flush()

            train_glb_batch_idx += 1

        # ======================== logging ========================
        if batch_idx % 100 == 0 and is_main_process():

            with open(train_log_path, 'a') as f:
                f.write(f"Epoch [{epoch}/{config['exp']['epochs']}]\n"
                        f"\tBatch {batch_idx}/{tot_iter}\n"
                        f"\tLoss Focal: {loss_focal.item():.4f}\n"
                        f"\tLoss Dice: {loss_dice.item():.4f}\n"
                        f"\tLR: {optimizer.param_groups[0]['lr']}\n"
                        f"\tGrad Norm: {grad_total_norm}\n")

    # ======================== checkpoint ========================
    if is_main_process():

        save_path = f"{train_ckpt_dir}/epoch_{epoch}.pth"

        save_on_master(
            {
                "model": predictor.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }, save_path)

        print(f"Save model to {save_path}")

    # ======================== scheduler ========================
    lr_scheduler.step()


def val_epoch(epoch, config, device, val_log_path, vggt, predictor, val_loader,
              direction):
    """
    Evaluate the predictor model for one validation epoch.

    Pipeline:
        1. Sample source mask points.
        2. Extract VGGT tracking features from source-target image pairs.
        3. Predict target masks using iterative refinement.
        4. Recover predictions to original image resolution.
        5. Compute segmentation IoU metrics.
        6. Aggregate distributed validation results across all GPUs.
        7. Save qualitative visualization results.

    Args:
        epoch (int):
            Current validation epoch.

        config (dict):
            Experiment configuration dictionary.

        device:
            Torch device.

        val_log_path (str):
            Path to validation log file.

        vggt:
            Frozen VGGT feature extractor.

        predictor (torch.nn.Module):
            Segmentation predictor model.

        val_loader:
            Validation dataloader.

        direction (str):
            Validation split identifier.
            Example:
                - "ego2exo"
                - "exo2ego"
    """

    predictor.eval()

    # Accumulated IoU statistics
    val_iou = 0.0
    val_iou_n = 0

    eval_iter = iter(val_loader)

    # ======================== validation loop ========================
    for batch_idx, batch in tqdm(enumerate(eval_iter),
                                 total=len(eval_iter),
                                 position=0,
                                 disable=not is_main_process(),
                                 dynamic_ncols=True):

        (src_img_b, src_mask_b, src_img_ori_b, src_mask_ori_b, tar_img_b,
         tar_mask_b, tar_img_ori_b, tar_mask_ori_b, tar_proj_points_b,
         tar_points_b, mask_cropped, meta) = batch

        with torch.no_grad():

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):

                # ========= VGGT feature extraction =========
                src_points_b, feat_map_b = vggt_feat(
                    vggt=vggt,
                    src_img_b=src_img_b,
                    src_mask_b=src_mask_b,
                    tar_img_b=tar_img_b,
                    vggt_sample_num=config['vggt']['sample_num'],
                    device=device)

                feat_map_b = feat_map_b.float()
                src_points_b = src_points_b.float()

                # Sample source point features
                src_points_feat_b = sample_feat_at_points(
                    feat_map_b[:, 0, ...], src_points_b,
                    config['data']['img_size'], config['data']['img_size'])

                src_points_feat_b = src_points_feat_b.float()

                # Ground-truth target points
                tar_points_b = torch.as_tensor(tar_points_b).float().squeeze(
                    1).to(device)  # [B, N, 2]

                # Initial refinement mask
                tar_input_mask_b = None

                # ============= iterative refinement ===============
                for _ in range(config['exp']['refine_time'] + 1):

                    (pred_mask_b, low_res_mask_b, pred_mask_recover_b,
                     pred_points_recover_b) = predictor(
                         feat_map_b=feat_map_b,
                         src_points_b=src_points_b,
                         tar_points_b=tar_points_b,
                         src_mask_b=src_mask_b,
                         src_points_feat_b=src_points_feat_b,
                         tar_mask_b=tar_input_mask_b,
                         ret_logits=False,
                         meta=meta,
                         tar_square=704)

                    tar_input_mask_b = low_res_mask_b.float()

            # Final prediction
            pred_mask_b = pred_mask_b[:, 0, ...]  # [B, H, W]

            B = len(src_img_b)

            # =============== metric computation ================
            for b in range(B):

                pred_mask = (
                    pred_mask_b[b].detach().cpu().numpy()[None, :].astype(
                        np.uint8))

                pred_mask_recover = (pred_mask_recover_b[b].detach().cpu().
                                     numpy()[None, :].astype(np.uint8))

                tar_mask_ori = tar_mask_ori_b[b][None, :]

                ious_sum, ious_n, ious = cal_iou(pred_mask_recover,
                                                 tar_mask_ori)

                val_iou += ious_sum
                val_iou_n += ious_n

                # =================== visualization ==================
                if (config['debug']['vis_batch'] - batch_idx > 0
                        and is_main_process()):

                    save_dir = (f"{CURRENT_DIR}/{config['exp']['exp_name']}"
                                f"/vis/{direction}/epoch_{epoch}")

                    os.makedirs(save_dir, exist_ok=True)

                    tar_points = (tar_points_b[b].detach().cpu().numpy())

                    src_points = (src_points_b[b].detach().cpu().numpy())

                    tar_recover_points = (pred_points_recover_b[b].float().
                                          detach().cpu().numpy())

                    # Cropped-resolution visualization
                    my_show_masks_group(
                        imgs=[src_img_b[b], tar_img_b[b], tar_img_b[b]],
                        maskss=[
                            src_mask_b[b][None, :], tar_mask_b[b][None, :],
                            pred_mask
                        ],
                        pointss=[
                            src_points[None, :], tar_points[None, :],
                            tar_points[None, :]
                        ],
                        save_dir=save_dir,
                        save_name=f'{batch_idx}_{b}_input_{ious[0]:.4f}.jpg')

                    # Original-resolution visualization
                    my_show_masks_group(
                        imgs=[
                            tar_img_ori_b[b], tar_img_ori_b[b],
                            tar_img_ori_b[b]
                        ],
                        maskss=[tar_mask_ori, tar_mask_ori, pred_mask_recover],
                        pointss=[
                            tar_proj_points_b[b][None, :],
                            tar_recover_points[None, :],
                            tar_recover_points[None, :]
                        ],
                        save_dir=save_dir,
                        save_name=f'{batch_idx}_{b}_ori_{ious[0]:.4f}.jpg')

            # ================ logging ================
            if batch_idx % 100 == 0 and is_main_process():

                with open(val_log_path, 'a') as f:
                    f.write(f"{direction}: \n\t"
                            f"Epoch [{epoch}/{config['exp']['epochs']}]: \n\t"
                            f"Batch [{batch_idx}/{len(val_loader)}] \n\t\t"
                            f"mIoU (local) "
                            f"{(val_iou / max(val_iou_n, 1)):.4f}\n")

    # =================== distributed aggregation ==============
    val_iou_tensor = torch.tensor([val_iou],
                                  dtype=torch.float32,
                                  device=device)

    val_iou_n_tensor = torch.tensor([val_iou_n],
                                    dtype=torch.float32,
                                    device=device)

    # Log local statistics before synchronization
    with open(val_log_path, 'a') as f:
        f.write(f"{direction}: \n\t"
                f"Epoch [{epoch}/{config['exp']['epochs']}]: \n\t"
                f"Local Rank: {local_rank} \n\t"
                f"Before all_reduce - all_iou: "
                f"{val_iou_tensor.item():.6f}, "
                f"all_n: {val_iou_n_tensor.item()}\n")

    torch.distributed.barrier()

    # Synchronize metrics across GPUs
    torch.distributed.all_reduce(val_iou_tensor,
                                 op=torch.distributed.ReduceOp.SUM)

    torch.distributed.all_reduce(val_iou_n_tensor,
                                 op=torch.distributed.ReduceOp.SUM)

    eps = 1e-6

    # ======================== final metric ========================
    if is_main_process():

        miou = (val_iou_tensor / (val_iou_n_tensor + eps)).cpu().item()

        with open(val_log_path, 'a') as f:
            f.write(f"{direction}: \n\t"
                    f"Epoch [{epoch}/{config['exp']['epochs']}]: \n\t"
                    f"After all_reduce - all_iou: "
                    f"{val_iou_tensor.item():.6f}, "
                    f"all_n: {val_iou_n_tensor.item()}\n\t"
                    f"[Final mIoU] {miou:.6f}\n")


if __name__ == '__main__':
    """
    Main training and evaluation entry for distributed VGGT-based
    ego-exo mask correspondence learning.

    Pipeline:
        1. Initialize distributed data parallel (DDP) environment.
        2. Load experiment configuration.
        3. Create logging / checkpoint directories.
        4. Load frozen VGGT feature extractor.
        5. Build predictor model.
        6. Initialize optimizer and LR scheduler.
        7. Resume training state if checkpoint exists.
        8. Build train / validation dataloaders.
        9. Run training + validation loops.
        10. Run full evaluation on validation benchmarks.

    Notes:
        - VGGT is frozen during training.
        - Only predictor parameters are optimized.
        - Supports multi-GPU distributed training with NCCL backend.
        - Validation is performed for both:
            * ego2exo
            * exo2ego
    """

    # ==================== DDP Initialization ====================
    # Initialize distributed training environment.

    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend='nccl')

    dist.barrier()

    setup_for_distributed(local_rank == 0)

    device = (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print('[NOTE] Finish DDP Setting')

    # =========================== Config Loading ===========================
    # Parse experiment configuration file.

    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='path to config file')

    args = parser.parse_args()

    config_path = args.config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = recursive_convert(config)

    print('[NOTE] Config Loaded')

    # =========================== Directory Setup ===========================
    # Create checkpoint and logging directories.

    train_ckpt_dir = (f"{CURRENT_DIR}/{config['exp']['exp_name']}/ckpts")

    os.makedirs(train_ckpt_dir, exist_ok=True)

    log_dir = (f"{CURRENT_DIR}/{config['exp']['exp_name']}/log")

    os.makedirs(log_dir, exist_ok=True)

    train_log_path = f'{log_dir}/train.txt'
    val_log_path = f'{log_dir}/val.txt'
    val_full_log_path = f'{log_dir}/val_full.txt'

    # =========================== VGGT Loading ===========================
    # Load frozen VGGT feature extractor.

    vggt = VGGT()

    vggt_ckpt = torch.load('../../third_party/vggt_main/model.pt',
                           map_location='cpu')

    vggt.load_state_dict(vggt_ckpt)

    vggt = vggt.to(device)

    del vggt_ckpt

    vggt.eval()

    print(f"[NOTE] VGGT Loaded on {device}")

    # ==================== Predictor Initialization ==================
    # Build trainable segmentation predictor.

    resume_ckpt = config['exp']['resume_ckpt']

    if config['exp']['resume_ckpt'] is not None:
        resume_ckpt = config['exp']['resume_ckpt']
    else:
        resume_ckpt = get_newest_ckpt(train_ckpt_dir)

    predictor = Predictor(img_size=config['data']['img_size'],
                          sa_tokens=config['data']['sa_tokens'],
                          resume_ckpt=resume_ckpt).to(device)

    predictor = DDP(predictor,
                    device_ids=[local_rank],
                    gradient_as_bucket_view=True)

    print("[NOTE] Predictor Loaded")

    # ================= Optimizer & Scheduler =====================
    # Initialize AdamW optimizer and MultiStep LR scheduler.

    params = predictor.parameters()

    optimizer = torch.optim.AdamW(params,
                                  lr=config['exp']['lr'],
                                  weight_decay=config['exp']['weight_decay'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config['exp']['lr_drops'], gamma=0.1)

    print("[NOTE] Optimizer Loaded")

    # =========================== Resume Training ===========================
    # Restore optimizer / scheduler states if resuming.

    start_epoch = 1

    if resume_ckpt is not None:

        ckpt = torch.load(resume_ckpt, map_location='cpu')

        # Restore optimizer state
        p_groups = copy.deepcopy(optimizer.param_groups)

        optimizer.load_state_dict(ckpt['optimizer'])

        # Preserve current LR configuration
        for pg, pg_old in zip(optimizer.param_groups, p_groups):
            pg['lr'] = pg_old['lr']
            pg['initial_lr'] = pg_old['initial_lr']

        # Restore scheduler state
        lr_scheduler.load_state_dict(ckpt['scheduler'])

        lr_scheduler.gamma = 0.1

        lr_scheduler.milestones = Counter(config['exp']['lr_drops'])

        lr_scheduler.step(ckpt['epoch'])

        # Resume epoch index
        last_epoch = ckpt['epoch']

        print(f"[NOTE] Using epoch {last_epoch}'s ckpt.")

        start_epoch = last_epoch + 1

        print(f'[NOTE] Starting epoch {start_epoch}')

    # ================== Dataloader Construction ===================
    # Build training and validation dataloaders.

    if config['exp']['mode'] == 'train':

        hybrid_train_loader = (Egoexo4dDataTrainDataLoader(config))

        ego2exo_val_loader, exo2ego_val_loader = (Egoexo4dDataValDataLoader(
            config, False))

    ego2exo_val_full_loader, exo2ego_val_full_loader = (
        Egoexo4dDataValDataLoader(config, True))

    # =========================== TensorBoard ===========================
    # Main process only.

    if is_main_process():
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # =========================== Training Stage ===========================
    if config['exp']['mode'] == 'train':

        print("[NOTE] Start Training...")

        # Initialize logs for fresh training
        if resume_ckpt is None:

            with open(train_log_path, 'w') as f:
                f.write('start:\n')

            with open(val_log_path, 'w') as f:
                f.write('start:\n')

        # =========================== Epoch Loop ===========================
        for epoch in range(start_epoch, config['exp']['epochs'] + 1):

            # Shuffle distributed sampler
            hybrid_train_loader.batch_sampler.sampler.set_epoch(epoch)

            # Train one epoch
            train_epoch(epoch, config, device, train_log_path, train_ckpt_dir,
                        vggt, predictor, optimizer, lr_scheduler,
                        hybrid_train_loader, writer)

            # ================= Validation ===================

            # ego2exo
            val_epoch(epoch,
                      config,
                      device,
                      val_log_path,
                      vggt,
                      predictor,
                      ego2exo_val_loader,
                      direction='ego2exo')

            # exo2ego
            val_epoch(epoch,
                      config,
                      device,
                      val_log_path,
                      vggt,
                      predictor,
                      exo2ego_val_loader,
                      direction='exo2ego')

    # =========================== Full Evaluation ===========================
    # Evaluate full validation benchmark.

    print("[NOTE] Starting Eval Full...")

    with open(val_full_log_path, 'w') as f:
        f.write('start:\n')

    # ego2exo
    val_epoch(13,
              config,
              device,
              val_full_log_path,
              vggt,
              predictor,
              ego2exo_val_full_loader,
              direction='ego2exo')

    # exo2ego
    val_epoch(13,
              config,
              device,
              val_full_log_path,
              vggt,
              predictor,
              exo2ego_val_full_loader,
              direction='exo2ego')

# =========================== Launch Examples ===========================

# 4-GPU distributed training
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.run \
#     --nproc_per_node=4 \
#     --master_port=29600 \
#     hybrid.py \
#     --config hybrid.yaml

# 2-GPU distributed training with config
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.run \
#     --nproc_per_node=2 \
#     --master_port=29600 \
#     hybrid.py \
#     --config hybrid.yaml
