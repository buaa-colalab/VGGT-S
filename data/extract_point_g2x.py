import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vggt.models.vggt import VGGT

from utils import mask_sampling, my_show_masks_group

# =========================================================
# Dataset
# =========================================================


class G2xCropImage(Dataset):

    def __init__(
        self,
        dataset_root,
        sample_path,
        img_size=518,
    ):

        self.dataset_root = dataset_root
        self.sample_path = sample_path
        self.img_size = img_size

        with open(self.sample_path, 'r') as fp:
            self.samples = json.load(fp)

        self.num_samples = len(self.samples)

        self.anns = self._load_mask_annotations()

        print(f"[NOTE] DataDir: "
              f"{self.sample_path} "
              f"with {self.num_samples} samples")

    def _load_mask_annotations(self):

        split_path = (f'{self.dataset_root}/split.json')

        with open(split_path, 'r') as fp:
            splits = json.load(fp)

        valid_takes = (splits['train'] + splits['val'])

        annotations = {}

        for take in valid_takes:

            mask_ann_path = (f'{self.dataset_root}/{take}/annotation.json')

            if os.path.exists(mask_ann_path):

                with open(mask_ann_path, 'r') as fp:
                    annotations[take] = json.load(fp)

        return annotations

    def pad_frame(self, frame, expect_size, is_mask):

        if not is_mask:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_NEAREST

        img = frame.copy()

        height, width = img.shape[:2]

        if width == height:

            img = cv2.resize(img, (expect_size[1], expect_size[0]),
                             interpolation=interp)

        else:

            if width > height:

                new_width = expect_size[1]

                new_height = round(height * (new_width / width) / 14) * 14

            else:

                new_height = expect_size[0]

                new_width = round(width * (new_height / height) / 14) * 14

            img = cv2.resize(img, (new_width, new_height),
                             interpolation=interp)

        h_padding = expect_size[0] - img.shape[0]
        w_padding = expect_size[1] - img.shape[1]

        if h_padding > 0 or w_padding > 0:

            if is_mask:

                img = np.pad(img,
                             pad_width=((0, h_padding), (0, w_padding)),
                             mode='constant',
                             constant_values=0)

            else:

                img = np.pad(img,
                             pad_width=((0, h_padding), (0, w_padding), (0,
                                                                         0)),
                             mode='constant',
                             constant_values=114)

        if is_mask:
            img = img.astype(np.uint8)

        return img

    def process_view(self, take_id, cam, obj, frame_idx, keep_ori):

        img_pth = (f'{self.dataset_root}/'
                   f'{take_id}/{cam}/{frame_idx}.jpg')

        img_ori = np.array(Image.open(img_pth).convert('RGB'))

        mask_ori = mask_utils.decode(
            self.anns[take_id]['masks'][obj][cam][frame_idx])

        h, w = mask_ori.shape

        if 'aria' in cam:

            binary_mask = cv2.resize(mask_ori, (w // 2, h // 2),
                                     interpolation=cv2.INTER_NEAREST)

        else:

            binary_mask = cv2.resize(mask_ori, (w // 4, h // 4),
                                     interpolation=cv2.INTER_NEAREST)

        if h != w:

            if keep_ori:

                img = img_ori
                mask = binary_mask

            else:

                img = self.pad_frame(img_ori, (self.img_size, self.img_size),
                                     is_mask=False)

                mask = self.pad_frame(binary_mask,
                                      (self.img_size, self.img_size),
                                      is_mask=True)

        else:

            img = cv2.resize(img_ori, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)

            mask = cv2.resize(binary_mask, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)

        assert int(mask.sum()) > 0

        return img, mask

    def __getitem__(
        self,
        idx,
    ):

        sample = self.samples[idx]

        take_id, ego_cam, exo_cam, obj, frame_idx = (sample.split('//'))

        frame_idx = str(int(frame_idx))

        ego_img, ego_mask = self.process_view(take_id=take_id,
                                              cam=ego_cam,
                                              obj=obj,
                                              frame_idx=frame_idx,
                                              keep_ori=False)

        exo_img, exo_mask = self.process_view(take_id=take_id,
                                              cam=exo_cam,
                                              obj=obj,
                                              frame_idx=frame_idx,
                                              keep_ori=False)

        exo_ori_img, exo_ori_mask = self.process_view(take_id=take_id,
                                                      cam=exo_cam,
                                                      obj=obj,
                                                      frame_idx=frame_idx,
                                                      keep_ori=True)

        return (ego_img, ego_mask, exo_img, exo_mask, exo_ori_img,
                exo_ori_mask, sample)

    def __len__(self):

        return self.num_samples


# =========================================================
# Collate
# =========================================================


def collate_image(batch):

    (src_img_b, src_mask_b, tar_img_b, tar_mask_b, tar_ori_img_b,
     tar_ori_mask_b, sample_b) = zip(*batch)

    src_img_b = np.stack(src_img_b)
    src_mask_b = np.stack(src_mask_b)

    tar_img_b = np.stack(tar_img_b)
    tar_mask_b = np.stack(tar_mask_b)

    return (src_img_b, src_mask_b, tar_img_b, tar_mask_b, tar_ori_img_b,
            tar_ori_mask_b, sample_b)


# =========================================================
# Loader
# =========================================================


def ImageLoader(
    batch_size,
    dataset_root,
    sample_path,
):

    train_set = G2xCropImage(
        dataset_root=dataset_root,
        sample_path=sample_path,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_image,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader


# =========================================================
# Main
# =========================================================


def main(
    loader,
    device,
    save_path,
    n_samples,
    vis_root,
    vis_batch_num,
):

    res = {}

    with torch.inference_mode():

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):

            for batch_idx, batch in tqdm(enumerate(loader),
                                         total=len(loader),
                                         position=0,
                                         dynamic_ncols=True):

                (src_img_b, src_mask_b, tar_img_b, tar_mask_b, tar_ori_img_b,
                 tar_ori_mask_b, sample_b) = batch

                src_points_b = mask_sampling(masks_b=src_mask_b[:, None, :],
                                             n_samples=n_samples)

                plot_src_points_b = src_points_b[:, 0]

                src_points_b = (torch.as_tensor(src_points_b).squeeze(
                    1).float().to(device))

                # =====================================================
                # Track
                # =====================================================

                src_img_tb = (torch.from_numpy(src_img_b / 255.0).permute(
                    0, 3, 1, 2).float())

                tar_img_tb = (torch.from_numpy(tar_img_b / 255.0).permute(
                    0, 3, 1, 2).float())

                img_pair_b = torch.stack([src_img_tb, tar_img_tb],
                                         dim=1).to(device).contiguous()

                aggregated_tokens_list, patch_start_idx = (
                    vggt.aggregator(img_pair_b))

                track_list, vis_list, conf_list = (vggt.track_head(
                    aggregated_tokens_list,
                    images=img_pair_b,
                    patch_start_idx=patch_start_idx,
                    query_points=src_points_b,
                ))

                track_list = track_list[-1]

                tar_points_b = (
                    track_list[:, 1, :, :].detach().clone().cpu().numpy())

                tar_first_points_b = np.clip(tar_points_b, 0, 517)

                conf_b = (conf_list[:, 1, ...].unsqueeze(
                    -1).float().detach().clone().cpu().numpy())

                vis_b = (vis_list[:, 1, ...].unsqueeze(
                    -1).float().detach().clone().cpu().numpy())

                tar_big_points_b = (tar_first_points_b * 960 / 518)

                B = len(src_img_b)

                for b in range(B):

                    val = {
                        'g2x_518_first': tar_first_points_b[b].tolist(),
                        'g2x_518_vis': vis_b[b].tolist(),
                        'g2x_518_conf': conf_b[b].tolist()
                    }

                    if sample_b[b] not in res:

                        res[sample_b[b]] = val

                        # =====================================================
                        # Visualization
                        # =====================================================

                        if batch_idx < vis_batch_num:

                            my_show_masks_group(
                                [src_img_b[b], tar_img_b[b], tar_ori_img_b[b]],
                                [
                                    src_mask_b[b][None, :],
                                    tar_mask_b[b][None, :],
                                    tar_ori_mask_b[b][None, :]
                                ],
                                vis_root,
                                f'{batch_idx}_{b}.jpg',
                                [
                                    plot_src_points_b[b][None, :],
                                    tar_first_points_b[b][None, :],
                                    tar_big_points_b[b][None, :]
                                ],
                                [None, None],
                            )

                # =====================================================
                # Save
                # =====================================================

                if batch_idx % 500 == 0:

                    with open(save_path, 'w', encoding='utf-8') as fp:

                        json.dump(res, fp, indent=4)

                    print(f'[NOTE] saved batch {batch_idx}')

    with open(save_path, 'w', encoding='utf-8') as fp:

        json.dump(res, fp, indent=4)

    print(f'[NOTE] final saved to {save_path}')


# =========================================================
# Entry
# =========================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='your/path/to/egoexo4d/dataset',
    )

    parser.add_argument(
        '--save_root',
        type=str,
        default='.',
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='val',
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
    )

    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=50,
    )

    parser.add_argument(
        '--vis_batch_num',
        type=int,
        default=10,
    )

    args = parser.parse_args()

    # =====================================================
    # Device
    # =====================================================

    local_rank = args.local_rank

    device = (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # =====================================================
    # Paths
    # =====================================================

    sample_path = f'{args.mode}_obj.json'

    save_path = os.path.join(args.save_root, f'{args.mode}_obj_wp_g2x.json')

    vis_root = os.path.join(args.save_root, 'vis')

    os.makedirs(vis_root, exist_ok=True)

    # =====================================================
    # Loader
    # =====================================================

    dataloader = ImageLoader(
        batch_size=args.batch_size,
        dataset_root=args.dataset_root,
        sample_path=sample_path,
    )

    # =====================================================
    # VGGT
    # =====================================================

    vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False)

    vggt_ckpt = torch.load('../third_party/vggt_main/model.pt',
                           map_location='cpu')

    vggt.load_state_dict(vggt_ckpt, strict=False)

    vggt = vggt.to(device)

    vggt.eval()

    del vggt_ckpt

    print("[NOTE] VGGT Loaded")

    # =====================================================
    # Run
    # =====================================================

    main(
        loader=dataloader,
        device=device,
        save_path=save_path,
        n_samples=args.n_samples,
        vis_root=vis_root,
        vis_batch_num=args.vis_batch_num,
    )
