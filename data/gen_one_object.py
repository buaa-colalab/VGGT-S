import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

# =========================================================
# Args
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='your/path/to/egoexo4d/dataset',
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='val',
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='.',
    )

    parser.add_argument(
        '--img_sizes',
        type=int,
        nargs='+',
        default=[518, 420, 700],
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
    )

    return parser.parse_args()


args = parse_args()

DATASET_ROOT = args.dataset_root
MODE = args.mode
IMG_SIZE_LIST = args.img_sizes

SCENE_PATH = f'./{MODE}_scenes.json'

SAVE_PATH = os.path.join(args.save_dir, f'{MODE}_obj.json')

# =========================================================
# Load annotations
# =========================================================


def load_mask_annotations(dataset_root):
    split_path = f'{dataset_root}/split.json'

    with open(split_path, 'r') as fp:
        splits = json.load(fp)

    valid_takes = splits['train'] + splits['val']

    annotations = {}

    for take in valid_takes:

        mask_ann_path = f'{dataset_root}/{take}/annotation.json'

        if os.path.exists(mask_ann_path):

            with open(mask_ann_path, 'r') as fp:
                annotations[take] = json.load(fp)

    return annotations


anns = load_mask_annotations(DATASET_ROOT)

# =========================================================
# Utils
# =========================================================


def pad_mask(mask, expect_size):

    interp = cv2.INTER_NEAREST

    img = mask.copy()

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

        img = cv2.resize(img, (new_width, new_height), interpolation=interp)

    h_padding = expect_size[0] - img.shape[0]
    w_padding = expect_size[1] - img.shape[1]

    if h_padding > 0 or w_padding > 0:

        img = np.pad(img,
                     pad_width=((0, h_padding), (0, w_padding)),
                     mode='constant',
                     constant_values=0)

    return img.astype(np.uint8)


def clip_exo_mean(exo_mask, tar_size):

    ori_h, ori_w = exo_mask.shape[:2]

    if ori_w >= ori_h:

        _, x_coords = np.where(exo_mask == 1)

        x_mid = int(x_coords.mean())

        left = min(max(x_mid - 270, 0), 420)
        right = left + 540

        clip_mask = cv2.resize(exo_mask[:, left:right], (tar_size, tar_size),
                               interpolation=cv2.INTER_NEAREST)

    else:

        y_coords, _ = np.where(exo_mask == 1)

        y_mid = int(y_coords.mean())

        top = min(max(y_mid - 270, 0), 420)
        bottom = top + 540

        clip_mask = cv2.resize(exo_mask[top:bottom, :], (tar_size, tar_size),
                               interpolation=cv2.INTER_NEAREST)

    return clip_mask


def load_binary_mask(
    take_id,
    cam,
    obj,
    frame_idx,
):

    mask_ori = mask_utils.decode(anns[take_id]['masks'][obj][cam][frame_idx])

    h, w = mask_ori.shape

    if 'aria' in cam:

        binary_mask = cv2.resize(mask_ori, (w // 2, h // 2),
                                 interpolation=cv2.INTER_NEAREST)

    else:

        binary_mask = cv2.resize(mask_ori, (w // 4, h // 4),
                                 interpolation=cv2.INTER_NEAREST)

    return binary_mask


def check_pad_valid(mask):

    padded = pad_mask(mask=mask, expect_size=(518, 518))

    return int(padded.sum()) > 0


def check_crop_valid(mask, tar_size):

    h, w = mask.shape

    if h != w:

        processed = clip_exo_mean(exo_mask=mask, tar_size=tar_size)

    else:

        processed = cv2.resize(mask, (tar_size, tar_size),
                               interpolation=cv2.INTER_NEAREST)

    return int(processed.sum()) > 0


def validate_view(
    take_id,
    cam,
    obj,
    frame_idx,
):

    if take_id not in anns:
        return False

    try:

        binary_mask = load_binary_mask(take_id, cam, obj, frame_idx)

    except Exception:

        return False

    if not check_pad_valid(binary_mask):
        return False

    for tar_size in IMG_SIZE_LIST:

        flag = check_crop_valid(binary_mask, tar_size)

        if not flag:
            return False

    return True


# =========================================================
# Worker
# =========================================================


def process_scene(scene):

    local_results = []

    for pair in scene:

        ego_pair, exo_pair = pair

        ego_rgb, ego_mask = ego_pair
        exo_rgb, exo_mask = exo_pair

        _, take_id, ego_cam, obj, _, idx = ego_rgb.split('//')
        _, take_id2, exo_cam, obj2, _, idx2 = exo_rgb.split('//')

        if (take_id != take_id2 or obj != obj2 or idx != idx2):
            continue

        idx = str(int(idx))

        ego_ok = validate_view(
            take_id=take_id,
            cam=ego_cam,
            obj=obj,
            frame_idx=idx,
        )

        if not ego_ok:
            continue

        exo_ok = validate_view(
            take_id=take_id,
            cam=exo_cam,
            obj=obj,
            frame_idx=idx,
        )

        if not exo_ok:
            continue

        idx_str = f"{int(idx):06d}"

        sample = '//'.join([take_id, ego_cam, exo_cam, obj, idx_str])

        local_results.append(sample)

    return local_results


# =========================================================
# Main
# =========================================================


def main():

    with open(SCENE_PATH, 'r') as fp:
        scenes = json.load(fp)

    results = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:

        futures = [executor.submit(process_scene, scene) for scene in scenes]

        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                dynamic_ncols=True,
        ):

            results.extend(future.result())

    print(f'[NOTE] valid samples: {len(results)}')

    with open(SAVE_PATH, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=4)

    print(f'[NOTE] saved to {SAVE_PATH}')


if __name__ == '__main__':
    main()
