import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset

from utils import (DistributedSampler, cal_crop_size_1_5_x, cal_crop_size_1_x,
                   cal_crop_size_2x, cal_crop_size_3x, clip_exo_box_center_wp,
                   clip_exo_mean, cluster_points, crop_and_resize_by_mask_safe,
                   remove_outliers_median)

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR.parent / "data"
EGOEXO4D_ROOT = "your/path/to/egoexo4d/dataset"  # Your Egoexo4d data path


class Egoexo4dData(Dataset):
    """
    Ego-exo correspondence dataset for training and evaluating
    cross-view mask propagation and object correspondence.

    This dataset supports:
        - ego -> exo correspondence
        - exo -> ego correspondence
        - dynamic target cropping
        - VGGT point-guided target localization
        - iterative mask refinement

    Main features:
        - Bidirectional training
        - Dynamic crop scaling
        - Robust retry mechanism
        - Distributed training support
        - Multi-view annotation loading

    Dataset structure:
        Each sample contains:
            - source image and mask
            - target image and mask
            - projected correspondence points
            - cropped-region metadata

    Args:
        mode (str):
            Dataset mode.
            Options:
                - 'train'
                - 'val'

        img_size (int):
            Input image resolution.

        step (int):
            Dataset subsampling interval.

        src_R (float):
            Source crop expansion ratio.

        direction (str, optional):
            Validation direction.
            Options:
                - 'ego2exo'
                - 'exo2ego'

        tar_num_points (int):
            Number of clustered target points.

        exo_tar_crop (float):
            Exocentric crop expansion ratio.

        linear_crop (int):
            Whether to use linear crop scaling.
    """

    def __init__(self,
                 mode,
                 img_size,
                 step,
                 src_R=2,
                 direction=None,
                 tar_num_points=5,
                 exo_tar_crop=3,
                 linear_crop=1):

        # Number of sampled target points
        self.tar_num_points = tar_num_points

        # Target crop scale factor
        self.exo_tar_crop = exo_tar_crop

        # Maximum retry times for invalid samples
        self.max_retry = 10

        # Whether to use linear crop strategy
        self.linear_crop = linear_crop

        # Dataset root directory
        self.dataset_root = EGOEXO4D_ROOT

        # Source crop expansion ratio
        self.src_R = src_R

        # Validation direction
        self.direction = direction

        # Dataset mode
        self.mode = mode

        # Annotation split file
        self.sample_path = (f'{DATA_DIR}/{mode}_obj_wp.json')

        # Input resolution
        self.img_size = img_size

        # ======================== load samples ========================

        with open(self.sample_path, 'r') as fp:
            cont = json.load(fp)

        self.samples = cont[::step]

        self.num_samples = len(self.samples)

        # Load segmentation annotations
        self.anns = self._load_mask_annotations()

        # Backup sample for retry fallback
        self.last_case = None

        print(f"[NOTE] DataDir: {self.sample_path} "
              f"with {self.num_samples} samples")

    def _load_mask_annotations(self):
        """
        Load mask annotations from all valid takes.

        Returns:
            dict:
                Nested annotation dictionary.
        """

        split_path = f'{EGOEXO4D_ROOT}/split.json'

        with open(split_path, 'r') as fp:
            splits = json.load(fp)

        valid_takes = (splits['train'] + splits['val'])

        annotations = {}

        for take in valid_takes:

            mask_ann_path = (self.dataset_root + '/' + take +
                             '/annotation.json')

            if os.path.exists(mask_ann_path):

                with open(mask_ann_path, 'r') as fp:
                    annotations[take] = json.load(fp)

        return annotations

    def exo_src(self, take_id, cam, obj, frame_idx, R=2):
        """
        Load exocentric source image and mask.

        Processing:
            1. Load image and mask
            2. Resize mask
            3. Crop around object
            4. Resize to model resolution

        Args:
            take_id (str):
                Take identifier.

            cam (str):
                Camera name.

            obj (str):
                Object identifier.

            frame_idx (str):
                Frame index.

            R (float):
                Crop expansion ratio.

        Returns:
            img:
                Cropped source image.

            mask:
                Cropped source mask.

            img_ori:
                Original image.

            binary_mask:
                Original resized mask.
        """

        img_pth = (self.dataset_root + '/' + take_id + '/' + cam + '/' +
                   frame_idx + '.jpg')

        img_ori = np.array(Image.open(img_pth).convert('RGB'))

        mask_ori = mask_utils.decode(
            self.anns[take_id]['masks'][obj][cam][frame_idx])

        h, w = mask_ori.shape

        binary_mask = cv2.resize(mask_ori, (w // 4, h // 4),
                                 interpolation=cv2.INTER_NEAREST)

        img_s, mask_s = clip_exo_mean(img_ori, binary_mask, self.img_size)

        img, mask = crop_and_resize_by_mask_safe(img_s, mask_s, R)

        # fallback if crop failed
        if mask.sum() == 0:
            return img_s, mask_s, img_ori, binary_mask

        return img, mask, img_ori, binary_mask

    def exo_tar(self,
                take_id,
                cam,
                obj,
                frame_idx,
                proj_points,
                conf,
                vis,
                tar_num=5):
        """
        Load exocentric target image using projected ego points.

        Pipeline:
            1. Load image and mask
            2. Remove outlier projections
            3. Estimate crop region
            4. Dynamically crop target region
            5. Cluster target points

        Args:
            proj_points:
                Projected correspondence points.

            conf:
                Point confidence.

            vis:
                Point visibility.

            tar_num (int):
                Number of clustered target points.

        Returns:
            img:
                Cropped target image.

            mask:
                Cropped target mask.

            img_ori:
                Original image.

            binary_mask:
                Original resized mask.

            proj_points:
                Refined projected points.

            tar_points:
                Clustered target points.

            mask_cropped:
                Whether GT mask is truncated.

            meta:
                Crop recovery metadata.
        """

        img_pth = (self.dataset_root + '/' + take_id + '/' + cam + '/' +
                   frame_idx + '.jpg')

        img_ori = np.array(Image.open(img_pth).convert('RGB'))

        mask_ori = mask_utils.decode(
            self.anns[take_id]['masks'][obj][cam][frame_idx])

        h, w = mask_ori.shape

        binary_mask = cv2.resize(mask_ori, (w // 4, h // 4),
                                 interpolation=cv2.INTER_NEAREST)

        # Clip projected points
        proj_points[:, 0] = np.clip(proj_points[:, 0], 0, w // 4 - 1)

        proj_points[:, 1] = np.clip(proj_points[:, 1], 0, h // 4 - 1)

        # Remove outliers
        proj_points, _ = remove_outliers_median(proj_points, K=10)

        # Bounding box
        x_min = np.min(proj_points[:, 0])
        y_min = np.min(proj_points[:, 1])
        x_max = np.max(proj_points[:, 0])
        y_max = np.max(proj_points[:, 1])

        box_size = max(y_max - y_min, x_max - x_min)

        # Dynamic crop size
        if self.exo_tar_crop == 3:
            crop_size = cal_crop_size_3x(box_size)

        elif self.exo_tar_crop == 2:
            crop_size = cal_crop_size_2x(box_size)

        elif self.exo_tar_crop == 1.5:

            if self.linear_crop == 1:
                crop_size = cal_crop_size_1_5_x(box_size)

            elif self.linear_crop == 0:
                crop_size = 360

            else:
                crop_size = None

        elif self.exo_tar_crop == 1:
            crop_size = cal_crop_size_1_x(box_size)

        # Crop target region
        img, mask, update_points, mask_cropped, meta = (clip_exo_box_center_wp(
            proj_points, img_ori, binary_mask, self.img_size, self.img_size,
            crop_size))

        # Cluster points
        _, tar_points = cluster_points(update_points, K=tar_num)

        return (img, mask, img_ori, binary_mask, proj_points, tar_points,
                mask_cropped, meta)

    def ego_src(self, take_id, cam, obj, frame_idx, R=2):
        """
        Load egocentric source image and mask.

        Returns:
            Cropped source image and masks.
        """

        img_pth = (self.dataset_root + '/' + take_id + '/' + cam + '/' +
                   frame_idx + '.jpg')

        img_ori = np.array(Image.open(img_pth).convert('RGB'))

        mask_ori = mask_utils.decode(
            self.anns[take_id]['masks'][obj][cam][frame_idx])

        h, w = mask_ori.shape

        binary_mask = cv2.resize(mask_ori, (w // 2, h // 2),
                                 interpolation=cv2.INTER_NEAREST)

        img_s = cv2.resize(img_ori, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_AREA)

        mask_s = cv2.resize(binary_mask, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_NEAREST)

        img, mask = crop_and_resize_by_mask_safe(img_s, mask_s, R)

        if mask.sum() == 0:
            return img_s, mask_s, img_ori, binary_mask

        return img, mask, img_ori, binary_mask

    def ego_tar(
        self,
        take_id,
        cam,
        obj,
        frame_idx,
    ):
        """
        Load egocentric target image and mask.

        Returns:
            Full-resolution resized ego image and mask.
        """

        img_pth = (self.dataset_root + '/' + take_id + '/' + cam + '/' +
                   frame_idx + '.jpg')

        img_ori = np.array(Image.open(img_pth).convert('RGB'))

        mask_ori = mask_utils.decode(
            self.anns[take_id]['masks'][obj][cam][frame_idx])

        h, w = mask_ori.shape

        binary_mask = cv2.resize(mask_ori, (w // 2, h // 2),
                                 interpolation=cv2.INTER_NEAREST)

        img = cv2.resize(img_ori, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)

        mask = cv2.resize(binary_mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        return img, mask, img_ori, binary_mask

    def __process__(self, idx):
        """
        Process a single sample.

        Supports:
            - ego -> exo
            - exo -> ego

        Returns:
            Full training/evaluation tuple.
        """

        sample = self.samples[idx]

        scene_info = sample['scene_info']

        take_id, ego_cam, exo_cam, obj, frame_idx = (scene_info.split('//'))

        frame_idx = str(int(frame_idx))

        exchange = False

        # Random bidirectional training
        if self.mode == 'train' and random.random() < 0.5:
            exchange = True

        # Validation direction
        elif self.mode == 'val' and self.direction == 'exo2ego':
            exchange = True

        meta = None

        # ============================================================
        # exo -> ego
        # ============================================================

        if exchange:

            src_img, src_mask, src_img_ori, src_mask_ori = (self.exo_src(
                take_id, exo_cam, obj, frame_idx, self.src_R))

            tar_img, tar_mask, tar_img_ori, tar_mask_ori = (self.ego_tar(
                take_id,
                ego_cam,
                obj,
                frame_idx,
            ))

            tar_points = np.array(sample['x2g_518'])

            tar_proj_points = tar_points * 704 / 518

            mask_cropped = False

        # ============================================================
        # ego -> exo
        # ============================================================

        else:

            src_img, src_mask, src_img_ori, src_mask_ori = (self.ego_src(
                take_id, ego_cam, obj, frame_idx, self.src_R))

            tar_points_50 = (np.array(sample['g2x_518_first']) * 960 / 518)

            (tar_img, tar_mask, tar_img_ori, tar_mask_ori, tar_proj_points,
             tar_points, mask_cropped, meta) = self.exo_tar(
                 take_id,
                 exo_cam,
                 obj,
                 frame_idx,
                 tar_points_50,
                 np.array(sample['g2x_518_conf']),
                 np.array(sample['g2x_518_vis']),
                 self.tar_num_points,
             )

        return (src_img, src_mask, src_img_ori, src_mask_ori, tar_img,
                tar_mask, tar_img_ori, tar_mask_ori, tar_proj_points,
                tar_points, mask_cropped, meta)

    def __getitem__(self, idx):
        """
        Load one dataset sample.

        Training:
            Retry invalid samples.

        Validation:
            Direct loading without retry.
        """

        # Validation
        if self.mode == 'val':
            return self.__process__(idx)

        # Retry invalid samples
        for retry in range(self.max_retry):

            idx2 = (idx + retry) % self.num_samples

            (src_img, src_mask, src_img_ori, src_mask_ori, tar_img, tar_mask,
             tar_img_ori, tar_mask_ori, tar_proj_points, tar_points,
             mask_cropped, meta) = self.__process__(idx2)

            # Valid sample check
            if (np.sum(tar_mask) > 0 and np.sum(src_mask)
                    >= self.img_size * self.img_size * 0.001):

                self.last_case = (src_img, src_mask, src_img_ori, src_mask_ori,
                                  tar_img, tar_mask, tar_img_ori, tar_mask_ori,
                                  tar_proj_points, tar_points, mask_cropped,
                                  meta)

                return self.last_case

        # Fallback
        if self.last_case is not None:
            return self.last_case

        return self.__process__(idx)

    def __len__(self):
        """
        Dataset size.
        """

        return self.num_samples


def Egoexo4dData_collate_fn(batch):
    """
    Custom collate function for ego-exo correspondence dataset.

    Args:
        batch:
            List of dataset samples.

    Returns:
        Batched numpy arrays and metadata.
    """

    (src_img_b, src_mask_b, src_img_ori_b, src_mask_ori_b, tar_img_b,
     tar_mask_b, tar_img_ori_b, tar_mask_ori_b, tar_proj_points_b,
     tar_points_b, mask_cropped, meta) = zip(*batch)

    src_img_b = np.stack(src_img_b)
    src_mask_b = np.stack(src_mask_b)

    tar_img_b = np.stack(tar_img_b)
    tar_mask_b = np.stack(tar_mask_b)

    tar_points_b = np.stack(tar_points_b)

    return (src_img_b, src_mask_b, src_img_ori_b, src_mask_ori_b, tar_img_b,
            tar_mask_b, tar_img_ori_b, tar_mask_ori_b, tar_proj_points_b,
            tar_points_b, mask_cropped, meta)


def Egoexo4dDataTrainDataLoader(config, ):
    """
    Build distributed training dataloader.

    Features:
        - DistributedSampler
        - Dynamic cropping
        - Bidirectional correspondence
        - Multi-worker loading

    Args:
        config:
            Experiment configuration.

    Returns:
        torch.utils.data.DataLoader
    """

    train_set = Egoexo4dData(
        mode='train',
        img_size=config['data']['img_size'],
        step=1,
        src_R=config['data']['src_R'],
        tar_num_points=config['vggt']['sample_num'],
        exo_tar_crop=config['data'].get('exo_tar_crop', 3),
        linear_crop=config['data'].get('linear_crop', 1),
    )

    sampler_train = DistributedSampler(train_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config['data']['train_batch_size'], drop_last=True)

    train_loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler_train,
        collate_fn=Egoexo4dData_collate_fn,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )

    return train_loader


def Egoexo4dDataValDataLoader(config, is_full):
    """
    Build validation dataloaders for:

        - ego2exo
        - exo2ego

    Args:
        config:
            Experiment configuration.

        is_full (bool):
            Whether to run full evaluation.

    Returns:
        tuple:
            (
                ego2exo_loader,
                exo2ego_loader
            )
    """

    # ============================================================
    # ego -> exo
    # ============================================================

    g2x_val_set = (Egoexo4dData(
        direction='ego2exo',
        mode='val',
        img_size=config['data']['img_size'],
        step=(config['data']['val_full_step']
              if is_full else config['data']['val_step']),
        src_R=config['data']['src_R'],
        tar_num_points=config['vggt']['sample_num'],
        exo_tar_crop=config['data'].get('exo_tar_crop', 3),
        linear_crop=config['data'].get('linear_crop', 1),
    ))

    g2x_sampler_val = DistributedSampler(g2x_val_set, shuffle=False)

    g2x_val_loader = DataLoader(
        g2x_val_set,
        config['data']['val_batch_size'],
        sampler=g2x_sampler_val,
        collate_fn=Egoexo4dData_collate_fn,
        drop_last=False,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )

    # ============================================================
    # exo -> ego
    # ============================================================

    x2g_val_set = (Egoexo4dData(
        direction='exo2ego',
        mode='val',
        img_size=config['data']['img_size'],
        step=(config['data']['val_full_step']
              if is_full else config['data']['val_step']),
        src_R=config['data']['src_R'],
        tar_num_points=config['vggt']['sample_num'],
        exo_tar_crop=config['data'].get('exo_tar_crop', 3),
        linear_crop=config['data'].get('linear_crop', 1),
    ))

    x2g_sampler_val = DistributedSampler(x2g_val_set, shuffle=False)

    x2g_val_loader = DataLoader(
        x2g_val_set,
        config['data']['val_batch_size'],
        sampler=x2g_sampler_val,
        collate_fn=Egoexo4dData_collate_fn,
        drop_last=False,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )

    return g2x_val_loader, x2g_val_loader
