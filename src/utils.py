"""
Some functions are modified from the SAM2 implementation.

Reference:
    SAM 2: Segment Anything in Images and Videos
    https://github.com/facebookresearch/sam2
"""

import ast
import builtins as __builtin__
import math
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data.sampler import Sampler

np.random.seed(42)


# ===========================  Config ===========================
def try_literal_eval(value):
    """
    Safely parse a string into a Python literal object.

    This function attempts to convert string representations of Python
    literals into their corresponding Python objects using
    `ast.literal_eval`. If parsing fails, the original value is returned.

    Supported conversions include:
        - "True"   -> True
        - "123"    -> 123
        - "[1,2]"  -> [1, 2]
        - "{'a':1}" -> {'a': 1}

    Args:
        value (Any):
            Input value to parse.

    Returns:
        Any:
            Parsed Python object if conversion succeeds,
            otherwise the original input value.
    """
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def recursive_convert(data):
    """
    Recursively apply `try_literal_eval` to nested data structures.

    This utility traverses dictionaries, lists, and tuples, converting
    string literals into their corresponding Python objects while
    preserving the original container structure.

    Commonly used for loading YAML/JSON configuration files where
    numeric, boolean, or list values may be stored as strings.

    Args:
        data (Any):
            Input nested structure.

    Returns:
        Any:
            Recursively converted structure with parsed literal values.
    """
    if isinstance(data, dict):
        return {k: recursive_convert(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_convert(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(recursive_convert(v) for v in data)
    else:
        return try_literal_eval(data)


# ===========================  Resume Search ===========================
def get_newest_ckpt(ckpt_dir):
    """
    Find the latest checkpoint file in a directory.

    This function recursively searches for checkpoint files matching
    the naming pattern:

        epoch_*.pth

    It extracts the epoch number from each filename, sorts all valid
    checkpoints by epoch index, and returns the checkpoint with the
    largest epoch number.

    Example:
        epoch_1.pth
        epoch_5.pth
        epoch_12.pth

    -> returns path to `epoch_12.pth`

    Args:
        ckpt_dir (str | Path):
            Root checkpoint directory.

    Returns:
        str | None:
            Absolute path to the newest checkpoint file.
            Returns None if no valid checkpoint is found.
    """
    ckpt_dir = Path(ckpt_dir)

    ckpt_files = []

    for path in ckpt_dir.rglob("epoch_*.pth"):
        match = re.search(r"epoch_(\d+)\.pth$", path.name)
        if match:
            epoch = int(match.group(1))
            ckpt_files.append((epoch, path))

    if len(ckpt_files) == 0:
        return None

    # sort by epoch number
    ckpt_files.sort(key=lambda x: x[0])

    newest_epoch, newest_path = ckpt_files[-1]

    return str(newest_path.resolve())


# ===========================  Visualization ===========================
def show_mask(mask, ax, random_color=False):
    """
    Overlay a segmentation mask on a matplotlib axis.

    The mask is visualized as a semi-transparent colored region.

    Args:
        mask (np.ndarray):
            Binary mask of shape (H, W).

        ax (matplotlib.axes.Axes):
            Target matplotlib axis.

        random_color (bool, optional):
            If True, use a random RGBA color.
            Otherwise, use a fixed color from matplotlib's `tab10`
            colormap. Default is False.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 2
        color = np.array([*cmap(cmap_idx)[:3], 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, ax, marker_size=200):
    """
    Visualize point prompts on a matplotlib axis.

    Points are displayed as green star markers.

    Args:
        coords (np.ndarray):
            Point coordinates of shape (N, 2),
            formatted as (x, y).

        ax (matplotlib.axes.Axes):
            Target matplotlib axis.

        marker_size (int, optional):
            Marker size for visualization.
            Default is 200.
    """
    ax.scatter(coords[:, 0],
               coords[:, 1],
               color='green',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=0.5)


def show_box(box, ax):
    """
    Draw a bounding box on a matplotlib axis.

    Args:
        box (np.ndarray | list):
            Bounding box represented by two corner points:
            [[x_min, y_min], [x_max, y_max]].

        ax (matplotlib.axes.Axes):
            Target matplotlib axis.
    """
    x0, y0 = box[0][0], box[0][1]
    w, h = box[1][0] - box[0][0], box[1][1] - box[0][1]

    ax.add_patch(
        plt.Rectangle((x0, y0),
                      w,
                      h,
                      edgecolor='green',
                      facecolor=(0, 0, 0, 0),
                      lw=2))


def my_show_masks(ax, image, masks, point_coords=None, bboxs=None):
    """
    Visualize an image with masks, points, and bounding boxes.

    Args:
        ax (matplotlib.axes.Axes):
            Target matplotlib axis.

        image (np.ndarray):
            RGB image of shape (H, W, 3).

        masks (list[np.ndarray] | None):
            List of binary masks to overlay.

        point_coords (list[np.ndarray] | None, optional):
            List of point coordinate arrays.

        bboxs (list[np.ndarray] | None, optional):
            List of bounding boxes.
    """
    ax.imshow(image)

    if masks is not None:
        for i, mask in enumerate(masks):
            show_mask(mask, ax)

    if point_coords is not None:
        for i, _ in enumerate(point_coords):
            show_points(point_coords[i], ax)

    if bboxs is not None:
        for i, _ in enumerate(bboxs):
            show_box(bboxs[i], ax)

    ax.axis('off')


def my_show_masks_group(
    imgs,
    maskss,
    save_dir,
    save_name,
    pointss=None,
    bboxss=None,
):
    """
    Visualize and save a group of images with corresponding masks.

    This function creates a horizontal visualization panel where each
    subplot contains:
        - an image,
        - optional segmentation masks,
        - optional point prompts,
        - optional bounding boxes.

    The figure size is automatically adjusted according to the image
    aspect ratio.

    Args:
        imgs (list[np.ndarray]):
            List of RGB images.

        maskss (list[list[np.ndarray] | None]):
            List of mask groups corresponding to each image.

        save_dir (str):
            Output directory for saving the visualization.

        save_name (str):
            Output filename.

        pointss (list[list[np.ndarray]] | None, optional):
            Point annotations for each image.

        bboxss (list[list[np.ndarray]] | None, optional):
            Bounding box annotations for each image.
            Only the first image typically visualizes boxes.

    Returns:
        None
    """
    # -------- 1. compute figure aspect ratio --------
    h, w = imgs[0].shape[:2]
    aspect = h / w

    n = len(imgs)

    # fixed width per image
    fig_width_per_img = 10
    fig_height = fig_width_per_img * aspect

    fig, axes = plt.subplots(1, n, figsize=(fig_width_per_img * n, fig_height))

    for i, ax in enumerate(axes):
        my_show_masks(
            ax,
            imgs[i],
            maskss[i] if maskss[i] is not None else None,
            point_coords=pointss[i] if pointss is not None else None,
            bboxs=bboxss[i] if bboxss is not None and i == 0 else None)

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    save_here = save_dir + '/' + save_name

    plt.savefig(save_here, dpi=300)
    print(f"saved in {save_here}")

    # release matplotlib memory
    plt.clf()
    plt.close('all')


# ===========================  Mask =========================
def kmeans_sampling(
        mask,  # H, W
        n_samples  # N
):
    """
    Sample representative foreground points from a binary mask
    using K-Means clustering.

    Args:
        mask (np.ndarray):
            Binary foreground mask of shape (H, W),
            where foreground pixels are 1.
        n_samples (int):
            Number of points to sample.

    Returns:
        np.ndarray:
            Sampled foreground points of shape (N, 2), in (x, y) format.

    Notes:
        - If the number of valid foreground pixels is smaller than `n_samples`,
          the sampled points are padded by repeating existing points.
        - Cluster centers that fall outside the foreground region are replaced
          with the nearest valid foreground point.
        - K-Means sampling provides spatially distributed points compared to
          random sampling.
    """
    valid_points = np.fliplr(np.argwhere(mask == 1).astype(np.float32))

    if len(valid_points) < n_samples:
        pad_num = n_samples - len(valid_points)
        padding = np.tile(valid_points[0], (pad_num, 1))
        selected = np.concatenate([valid_points, padding])
        return selected

    kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=1)
    kmeans.fit(valid_points)

    centers = kmeans.cluster_centers_

    samples = set()

    for center in centers:
        x, y = int(round(center[0])), int(round(center[1]))
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y,
                                                                      x] > 0:
            samples.add((x, y))
        else:
            distances = np.linalg.norm(valid_points - center, axis=1)
            sorted_idx = np.argsort(distances)
            for idx in sorted_idx:
                candidate = tuple(valid_points[idx].astype(int))
                if candidate not in samples:
                    samples.add(candidate)
                    break

    samples = np.array(list(samples), dtype=np.float32)

    if len(samples) < n_samples:
        pad_num = n_samples - len(samples)
        padding = np.tile(samples[0], (pad_num, 1))
        samples = np.concatenate([samples, padding])

    return samples


def mask_sampling(
        masks_b,  # B, O, H, W
        n_samples):
    """
    Perform point sampling for a batch of masks.

    For each object mask in the batch, representative foreground points are
    sampled using K-Means clustering.

    Args:
        masks_b (np.ndarray):
            Batch of binary masks with shape (B, O, H, W), where:
                - B: batch size
                - O: number of objects
                - H, W: spatial resolution
        n_samples (int):
            Number of sampled points per object mask.

    Returns:
        np.ndarray:
            Sampled point coordinates of shape (B, O, N, 2),
            where each point is represented as (x, y).

    Notes:
        - Empty masks are filled with zero coordinates.
        - Memory for the output tensor is pre-allocated for efficiency.
    """
    B, O, _, _ = masks_b.shape

    # pre-allocate memory for efficiency
    samples_b = np.zeros((B, O, n_samples, 2), dtype=np.float32)

    for b in range(B):
        for o in range(O):
            if int(masks_b[b, o].sum()) > 0:
                corrds = kmeans_sampling(masks_b[b, o], n_samples=n_samples)
            else:
                corrds = np.zeros((n_samples, 2), dtype=np.float32)

            samples_b[b, o] = corrds

    return samples_b  # B, O, N, 2


def cal_iou(
        pred_masks_b,  # B, H, W
        gt_b,  # B, H, W
):
    """
    Compute Intersection-over-Union (IoU) between predicted masks
    and ground truth masks.

    Args:
        pred_masks_b (np.ndarray):
            Predicted binary masks of shape (O, H, W).
        gt_b (np.ndarray):
            Ground-truth binary masks of shape (O, H, W).

    Returns:
        Tuple[float, int, np.ndarray]:
            - Sum of IoUs across all objects.
            - Number of evaluated objects.
            - Per-object IoU values of shape (O,).

    Notes:
        IoU is computed as:

            IoU = intersection / union

        where:
            intersection = pred ∩ gt
            union = pred ∪ gt

        A small epsilon is added to avoid division by zero.
    """
    O, _, _ = gt_b.shape
    eps = 1e-6

    gt_flat_b = gt_b.reshape(O, -1).astype(np.float32)
    pred_masks_flat_b = pred_masks_b.reshape(O, -1).astype(np.float32)

    intersect_b = (gt_flat_b * pred_masks_flat_b).sum(axis=-1)  # O
    gt_sum_b = gt_flat_b.sum(axis=-1)  # O
    pred_masks_sum_b = pred_masks_flat_b.sum(axis=-1)  # O
    union_b = gt_sum_b + pred_masks_sum_b - intersect_b  # O

    ious_b = intersect_b / (union_b + eps)  # O

    return ious_b.sum(), len(ious_b), ious_b


# ===========================  DistributedSampler ===========================
class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 local_rank=None,
                 local_size=None,
                 shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        print(f'call set_epoch: {epoch}')
        self.epoch = epoch


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_main_process():
    return get_rank() == 0


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


# ===========================  Loss ===========================
def dice_loss(
    pred_masks,
    gts,
    wanna_sum=False,
    wanna_list=False,
):
    """
    Compute Dice loss for binary mask prediction.

    Dice loss measures the overlap between predicted masks
    and ground-truth masks,
    and is commonly used in segmentation tasks to optimize region similarity.

    The Dice coefficient is defined as:

    :contentReference[oaicite:0]{index=0}

    and the loss is:

    :contentReference[oaicite:1]{index=1}

    Args:
        pred_masks (torch.Tensor):
            Predicted mask logits of shape (B, H, W) or (B, 1, H, W).
        gts (torch.Tensor):
            Ground-truth binary masks with the same shape as `pred_masks`.
        wanna_sum (bool, optional):
            If True, return the summed loss across the batch.
        wanna_list (bool, optional):
            If True, return per-sample Dice losses.

    Returns:
        torch.Tensor:
            - Mean Dice loss (default)
            - Summed Dice loss if `wanna_sum=True`
            - Per-sample Dice loss if `wanna_list=True`

    Notes:
        - Sigmoid activation is applied internally to prediction logits.
        - A smoothing factor (+1) is added to numerator and denominator
          to improve numerical stability and avoid division by zero.
    """
    pred_masks = pred_masks.sigmoid()
    pred_masks = pred_masks.flatten(start_dim=1)
    gts = gts.flatten(start_dim=1)

    numerator = 2 * (pred_masks * gts).sum(-1)
    denominator = pred_masks.sum(-1) + gts.sum(-1)

    loss = 1 - (numerator + 1) / (denominator + 1)

    if wanna_list:
        return loss

    if wanna_sum:
        return loss.sum()

    return loss.mean()


def focal_loss(pred_masks, gts, wanna_sum=False, wanna_list=False):
    """
    Compute binary Focal loss for segmentation.

    Focal loss down-weights easy examples and focuses training on hard samples,
    which is especially useful for highly imbalanced foreground/background
    segmentation tasks.

    The focal loss formulation is:

    :contentReference[oaicite:2]{index=2}

    where:
        - p_t is the predicted probability of the target class
        - γ (gamma) controls hard-example focusing
        - α balances positive and negative samples

    In this implementation:
        - γ = 2
        - α = 0.25

    Args:
        pred_masks (torch.Tensor):
            Predicted mask logits of shape (B, H, W) or (B, 1, H, W).
        gts (torch.Tensor):
            Ground-truth binary masks with the same shape as `pred_masks`.
        wanna_sum (bool, optional):
            If True, return summed loss over the batch.
        wanna_list (bool, optional):
            If True, return per-sample losses.

    Returns:
        torch.Tensor:
            - Mean focal loss (default)
            - Summed focal loss if `wanna_sum=True`
            - Per-sample focal loss if `wanna_list=True`

    Notes:
        - The final loss is averaged spatially for each sample.
    """
    prob = pred_masks.sigmoid()

    ce_loss = F.binary_cross_entropy_with_logits(pred_masks,
                                                 gts,
                                                 reduction='none')

    p_t = prob * gts + (1 - prob) * (1 - gts)

    loss = ce_loss * ((1 - p_t)**2)

    alpha = 0.25
    alpha_t = alpha * gts + (1 - alpha) * (1 - gts)

    loss = alpha_t * loss

    per_sample_loss = loss.mean(dim=[1, 2])

    if wanna_list:
        return per_sample_loss

    if wanna_sum:
        return per_sample_loss.sum()

    return per_sample_loss.mean()


# ===========================  Point ===========================
def recover_square_crop_tensor(mask_resized, points_resized, meta):
    """
    Recover a resized square-cropped mask and point coordinates back to
    the original image coordinate system.

    This function reverses the preprocessing pipeline:
        original image
            -> square crop
            -> resize
            -> prediction
            -> recover to original resolution

    Args:
        mask_resized (torch.Tensor):
            Predicted mask in resized crop space.
            Supported shapes:
                - (H, W)
                - (1, H, W)
                - (1, 1, H, W)

        points_resized (torch.Tensor):
            Point coordinates in resized crop space with shape (N, 2),
            represented as (x, y).

        meta (dict):
            Metadata describing the crop operation:
                {
                    "x1", "y1", "x2", "y2":
                        crop region in original image,

                    "ori_h", "ori_w":
                        original image resolution,

                    "crop_size":
                        square crop size before resizing,

                    "H", "W":
                        resized output resolution
                }

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            restored_mask:
                Recovered mask of shape (ori_h, ori_w)
                in the original image coordinate system.

            restored_points:
                Recovered point coordinates of shape (N, 2)
                in the original image coordinate system.

    Notes:
        Recovery procedure:
            1. Resize prediction back to crop resolution.
            2. Paste crop prediction into original image canvas.
            3. Map resized coordinates back to crop space.
            4. Shift coordinates into original image space.

        Regions outside the crop are initialized to -32.0,
        which is consistent with the mask logit representation.
    """

    device = mask_resized.device
    dtype = mask_resized.dtype

    # =========================
    # 1. unify mask shape
    # =========================
    mask = mask_resized

    if mask.dim() == 4:
        mask = mask[0, 0]
    elif mask.dim() == 3:
        mask = mask[0]
    elif mask.dim() != 2:
        raise ValueError("mask_resized must be (H,W), (1,H,W), or (1,1,H,W)")

    # =========================
    # 2. load crop metadata
    # =========================
    x1 = meta["x1"]
    y1 = meta["y1"]
    x2 = meta["x2"]
    y2 = meta["y2"]

    ori_h = meta["ori_h"]
    ori_w = meta["ori_w"]

    crop_size = meta["crop_size"]

    H_out = meta["H"]
    W_out = meta["W"]

    # =========================
    # 3. resize prediction back
    #    to crop resolution
    # =========================
    mask_crop = F.interpolate(mask[None, None].float(),
                              size=(crop_size, crop_size),
                              mode="bilinear",
                              align_corners=False)[0, 0]

    # =========================
    # 4. paste crop prediction
    #    into original canvas
    # =========================
    restored_mask = torch.full((ori_h, ori_w),
                               fill_value=-32.0,
                               device=device,
                               dtype=dtype)

    restored_mask[y1:y2, x1:x2] = mask_crop

    # =========================
    # 5. recover point coordinates
    # =========================
    restored_points = None

    if points_resized is not None and points_resized.numel() > 0:

        points = points_resized.to(device=device, dtype=dtype)

        # resized space -> crop space
        scale_x = crop_size / float(W_out)
        scale_y = crop_size / float(H_out)

        x_crop = points[:, 0] * scale_x
        y_crop = points[:, 1] * scale_y

        # crop space -> original image space
        x_orig = x_crop + x1
        y_orig = y_crop + y1

        restored_points = torch.stack([x_orig, y_orig], dim=1)

    return restored_mask, restored_points


def sample_feat_at_points(feat, points, img_H, img_W):
    """
    Sample image feature vectors at specified point coordinates
    using bilinear interpolation.

    This function extracts point-wise features from a dense feature map
    using differentiable grid sampling.

    Args:
        feat (torch.Tensor):
            Dense feature map of shape (B, C, H, W).

        points (torch.Tensor):
            Point coordinates of shape (B, N, 2),
            represented as (x, y) in image coordinates.

        img_H (int):
            Original image height used for coordinate normalization.

        img_W (int):
            Original image width used for coordinate normalization.

    Returns:
        torch.Tensor:
            Sampled point features of shape (B, N, C).

    Notes:
        Coordinate conversion:
            PyTorch `grid_sample` expects normalized coordinates in [-1, 1]:

        :contentReference[oaicite:0]{index=0}

        :contentReference[oaicite:1]{index=1}

        Additional details:
            - Bilinear interpolation is used for smooth feature sampling.
            - Border padding avoids invalid sampling outside image boundaries.
            - Returned features preserve differentiability for backpropagation.
    """
    B, C, H, W = feat.shape
    _, N, _ = points.shape

    points = points.float()

    x = points[..., 0]
    y = points[..., 1]

    # normalize coordinates to [-1, 1]
    x_norm = 2 * x / (img_W - 1) - 1
    y_norm = 2 * y / (img_H - 1) - 1

    # clamp to valid range
    x_norm = x_norm.clamp(-1, 1)
    y_norm = y_norm.clamp(-1, 1)

    # [B, N, 1, 2]
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(2)

    sampled = F.grid_sample(feat,
                            grid,
                            mode='bilinear',
                            padding_mode='border',
                            align_corners=True)

    # [B, C, N, 1] -> [B, N, C]
    feat_points = sampled.squeeze(-1).permute(0, 2, 1)

    return feat_points.float()


# ===========================  Dataloader ===========================
def clip_exo_box_center_wp(points,
                           tar_ori_img,
                           tar_ori_mask,
                           H,
                           W,
                           crop_size=540):
    """
    Perform square cropping centered on the bounding box of projected points,
    then resize the cropped region to a fixed resolution.

    The crop region is determined by the center of the point bounding box,
    while ensuring the crop window remains fully inside the original image.

    Processing pipeline:
        1. Compute bounding box from projected points.
        2. Compute bbox center.
        3. Construct square crop window.
        4. Clamp crop window inside image boundaries.
        5. Crop image and mask.
        6. Resize cropped region to target resolution.
        7. Remap point coordinates into resized crop space.
        8. Record metadata for future recovery.

    Args:
        points (np.ndarray):
            Projected point coordinates of shape (N, 2),
            represented as (x, y).

        tar_ori_img (np.ndarray):
            Original RGB image of shape (H_ori, W_ori, 3).

        tar_ori_mask (np.ndarray):
            Original binary mask of shape (H_ori, W_ori).

        H (int):
            Target output height after resizing.

        W (int):
            Target output width after resizing.

        crop_size (int, optional):
            Square crop size in original image space.
            Default: 540.

    Returns:
        Tuple:
            clip_img (np.ndarray):
                Cropped and resized RGB image of shape (H, W, 3).

            clip_mask (np.ndarray):
                Cropped and resized binary mask of shape (H, W).

            clip_points (np.ndarray):
                Remapped point coordinates in resized crop space,
                shape (N, 2).

            mask_cropped (bool):
                Whether part of the foreground mask lies outside
                the crop region.

            meta (dict):
                Metadata for crop recovery:
                    {
                        "x1", "y1", "x2", "y2":
                            crop coordinates in original image,

                        "crop_size":
                            original square crop size,

                        "ori_h", "ori_w":
                            original image resolution,

                        "H", "W":
                            resized output resolution
                    }

    Notes:
        Coordinate remapping from crop space to resized space:

        Additional details:
            - The crop window is always square.
            - No padding is used.
            - Coordinates are clipped into valid image bounds.
            - Metadata can later be used to recover predictions
              back to the original image space.
    """
    crop_size = int(crop_size)

    ori_h, ori_w = tar_ori_img.shape[:2]
    half_crop = crop_size / 2.0

    # ---------- 1. compute bbox ----------
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    # ---------- 2. compute bbox center ----------
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    # ---------- 3. raw crop window ----------
    x1 = round(cx - half_crop)
    y1 = round(cy - half_crop)

    # ---------- 4. clamp crop window ----------
    x1 = int(np.clip(x1, 0, ori_w - crop_size))
    y1 = int(np.clip(y1, 0, ori_h - crop_size))

    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # ---------- 5. check whether mask is cropped ----------
    mask_coords = np.argwhere(tar_ori_mask > 0)

    if mask_coords.shape[0] == 0:
        mask_cropped = True
    else:
        mask_xmin = np.min(mask_coords[:, 1])
        mask_ymin = np.min(mask_coords[:, 0])
        mask_xmax = np.max(mask_coords[:, 1])
        mask_ymax = np.max(mask_coords[:, 0])

        mask_cropped = not (mask_xmin >= x1 and mask_ymin >= y1
                            and mask_xmax <= x2 and mask_ymax <= y2)

    # ---------- 6. crop ----------
    crop_img = tar_ori_img[y1:y2, x1:x2]
    crop_mask = tar_ori_mask[y1:y2, x1:x2]

    # ---------- 7. resize ----------
    clip_img = cv2.resize(crop_img, (W, H), interpolation=cv2.INTER_AREA)

    clip_mask = cv2.resize(crop_mask.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST)

    # ---------- 8. remap points ----------
    scale_x = W / crop_size
    scale_y = H / crop_size

    clip_points = points.copy()

    clip_points[:, 0] = (clip_points[:, 0] - x1) * scale_x
    clip_points[:, 1] = (clip_points[:, 1] - y1) * scale_y

    clip_points[:, 0] = np.clip(clip_points[:, 0], 0, W - 1)
    clip_points[:, 1] = np.clip(clip_points[:, 1], 0, H - 1)

    # ---------- 9. recovery metadata ----------
    meta = {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "crop_size": int(crop_size),
        "ori_h": int(ori_h),
        "ori_w": int(ori_w),
        "H": int(H),
        "W": int(W),
    }

    return (clip_img, clip_mask, clip_points.astype(np.float32), mask_cropped,
            meta)


def clip_exo_mean(
    exo_img,
    exo_mask,
    tar_size,
):
    """
    Crop an egocentric/exocentric image around the foreground mask center,
    then resize it to a square target resolution.

    The crop region is determined using the mean foreground position
    along the longer image dimension.

    Args:
        exo_img (np.ndarray):
            Input RGB image of shape (H, W, 3).

        exo_mask (np.ndarray):
            Binary foreground mask of shape (H, W).

        tar_size (int):
            Output square resolution.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            clip_img:
                Resized cropped RGB image of shape
                (tar_size, tar_size, 3).

            clip_mask:
                Resized cropped binary mask of shape
                (tar_size, tar_size).

    Notes:
        Cropping strategy:
            - If image width >= height:
                crop horizontally around foreground center.
            - Otherwise:
                crop vertically around foreground center.

        The crop size is fixed to 540 pixels before resizing.

        Cropped coordinates are constrained inside valid image bounds
        to avoid padding artifacts.
    """
    ori_h, ori_w = exo_img.shape[:2]

    # horizontal crop
    if ori_w >= ori_h:

        _, x_coords = np.where(exo_mask == 1)

        x_mid = int(x_coords.mean())

        left = min(max(x_mid - 270, 0), 420)
        right = left + 540

        clip_img = cv2.resize(exo_img[:, left:right, :], (tar_size, tar_size),
                              interpolation=cv2.INTER_AREA)

        clip_mask = cv2.resize(exo_mask[:, left:right], (tar_size, tar_size),
                               interpolation=cv2.INTER_NEAREST)

    # vertical crop
    else:

        y_coords, _ = np.where(exo_mask == 1)

        y_mid = int(y_coords.mean())

        top = min(max(y_mid - 270, 0), 420)
        bottom = top + 540

        clip_img = cv2.resize(exo_img[top:bottom, :, :], (tar_size, tar_size),
                              interpolation=cv2.INTER_AREA)

        clip_mask = cv2.resize(exo_mask[top:bottom, :], (tar_size, tar_size),
                               interpolation=cv2.INTER_NEAREST)

    return clip_img, clip_mask


def cal_crop_size_3x(box_size, ):
    """
    Compute dynamic crop size with an approximately 3× object context ratio.

    The crop size is adaptively adjusted according to
    the target bounding-box size:
        - Small objects use a minimum crop size for sufficient context.
        - Large objects use the maximum crop size.
        - Intermediate objects are linearly interpolated.

    Args:
        box_size:
            Scalar bounding-box size estimated from projected target points.

    Returns:
        Adaptive square crop size.
    """
    if box_size < 50:
        return 180
    elif box_size > 200:
        return 540
    else:
        return int(2.4 * (box_size - 50) + 180)


def cal_crop_size_2x(box_size, ):
    """
    Compute dynamic crop size with an approximately 2× object context ratio.

    The crop region scales linearly with object size while remaining within
    predefined minimum and maximum crop bounds.

    Args:
        box_size:
            Scalar bounding-box size estimated from projected target points.

    Returns:
        Adaptive square crop size.
    """
    if box_size < 80:
        return 270
    elif box_size > 200:
        return 540
    else:
        return int(2.25 * (box_size - 80) + 270)


def cal_crop_size_1_5_x(box_size, ):
    """
    Compute dynamic crop size with an approximately 1.5× object context ratio.

    Compared with larger-context strategies, this setting produces
    tighter crops around the target object
    while still preserving limited surrounding context.

    Args:
        box_size:
            Scalar bounding-box size estimated from projected target points.

    Returns:
        Adaptive square crop size.
    """
    if box_size < 100:
        return 360
    elif box_size > 200:
        return 540
    else:
        return int(1.8 * (box_size - 100) + 360)


def cal_crop_size_1_x(box_size, ):
    """
    Return a fixed crop size without additional dynamic scaling.

    This strategy keeps the crop region constant regardless of object scale.

    Args:
        box_size:
            Unused placeholder for interface consistency.

    Returns:
        Fixed square crop size.
    """
    return 540


def remove_outliers_median(points, K=10):
    """
    Remove outlier points based on distance to the median center.

    A robust median center is first computed from all input points.
    Points with the largest Euclidean distances to the median center
    are treated as outliers.

    Args:
        points:
            Array of shape (N, 2) representing 2D coordinates.
        K:
            Number of outlier points to remove.

    Returns:
        keep_points:
            Remaining inlier points with shape (N-K, 2).
        outliers:
            Removed outlier points with shape (K, 2).
    """

    points = np.asarray(points)

    # 1. robust center (median)
    center = np.median(points, axis=0)

    # 2. distance to median
    dist = np.linalg.norm(points - center, axis=1)

    # 3. sort by distance (farthest first)
    idx = np.argsort(-dist)

    outlier_idx = idx[:K]
    keep_idx = idx[K:]

    return points[keep_idx], points[outlier_idx]


def crop_and_resize_by_mask_safe(img, mask, R):
    """
    Crop image regions around the foreground mask while ensuring
    that the complete mask remains inside the crop.

    The crop size is dynamically adjusted according to the mask
    bounding box and then resized back to the original resolution.

    Args:
        img:
            Input RGB image of shape (H, W, 3).
        mask:
            Binary foreground mask of shape (H, W).
        R:
            Crop ratio controlling the target crop size.

    Returns:
        img_resized:
            Cropped and resized image.
        mask_resized:
            Cropped and resized binary mask.
    """
    H, W = mask.shape

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        # fallback：center crop
        crop_size = int(H / R)
        cx, cy = W // 2, H // 2
        half = crop_size // 2
        x1 = max(0, min(cx - half, W - crop_size))
        y1 = max(0, min(cy - half, H - crop_size))
        x2 = x1 + crop_size
        y2 = y1 + crop_size
    else:
        # -------- 1. mask bounding box --------
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        bw = x_max - x_min + 1
        bh = y_max - y_min + 1

        # -------- 2. target crop size --------
        target_size = int(H / R)

        # -------- 3. ensure mask fully covered --------
        crop_size = max(target_size, bw, bh)

        # -------- 4. expand around mask center --------
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        half = crop_size // 2

        x1 = cx - half
        y1 = cy - half
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # -------- 5. boundary correction --------
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > W:
            shift = x2 - W
            x1 -= shift
            x2 = W
        if y2 > H:
            shift = y2 - H
            y1 -= shift
            y2 = H

        # final clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

    # -------- 6. crop --------
    mask_crop = mask[y1:y2, x1:x2]
    img_crop = img[y1:y2, x1:x2]

    # -------- 7. resize back --------
    mask_resized = cv2.resize(mask_crop, (W, H),
                              interpolation=cv2.INTER_NEAREST)
    img_resized = cv2.resize(img_crop, (W, H), interpolation=cv2.INTER_AREA)

    return img_resized, mask_resized


def cluster_points(points, K):
    """
    Cluster 2D points using K-Means.

    The function partitions input points into K clusters and returns
    both cluster assignments and cluster centers.

    Args:
        points:
            Array of shape (N, 2) containing 2D coordinates.
        K:
            Number of clusters.

    Returns:
        labels:
            Cluster assignment for each input point, shape (N,).
        centers:
            Cluster center coordinates, shape (K, 2).
    """
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto")
    kmeans.fit(points)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return labels, centers
