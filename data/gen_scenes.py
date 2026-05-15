import argparse
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root',
        type=str,
        default='your/path/to/egoexo4d/dataset',
    )

    parser.add_argument('--mode',
                        type=str,
                        choices=['train', 'val'],
                        default='train')

    parser.add_argument('--save_path', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    root = args.root
    mode = args.mode

    if args.save_path is None:
        save_path = f'{mode}_scenes.json'
    else:
        save_path = args.save_path

    # ---------------------------------------------------------
    # load g2x
    # ---------------------------------------------------------
    g2x = []

    with open(f'{root}/{mode}_exoego_pairs.json', 'r') as fp:
        cont = json.load(fp)

        for c in cont:
            g_r, g_m, x_r, x_m = c

            _, s_id, g_cam, obj, _, index = g_r.split('//')
            _, s_id, x_cam, obj, _, index = x_r.split('//')

            item = '//'.join([s_id, g_cam, x_cam, obj, index])

            g2x.append(item)

    # ---------------------------------------------------------
    # load x2g
    # ---------------------------------------------------------
    x2g = []

    with open(f'{root}/{mode}_egoexo_pairs.json', 'r') as fp:
        cont = json.load(fp)

        for c in cont:
            g_r, g_m, x_r, x_m = c

            _, s_id, g_cam, obj, _, index = g_r.split('//')
            _, s_id, x_cam, obj, _, index = x_r.split('//')

            item = '//'.join([s_id, g_cam, x_cam, obj, index])

            x2g.append(item)

    # ---------------------------------------------------------
    # intersection
    # ---------------------------------------------------------
    intersection = list(set(x2g) & set(g2x))

    print(f'[INFO] intersection num: {len(intersection)}')

    # ---------------------------------------------------------
    # organize by scene
    # ---------------------------------------------------------
    scene_dict = defaultdict(list)

    for item in intersection:
        s_id, g_cam, x_cam, obj, index = item.split('//')

        ego_rgb = ('downsampled_data//' +
                   '//'.join([s_id, g_cam, obj, 'rgb', index]))

        ego_mask = ('downsampled_data//' +
                    '//'.join([s_id, g_cam, obj, 'mask', index]))

        exo_rgb = ('downsampled_data//' +
                   '//'.join([s_id, x_cam, obj, 'rgb', index]))

        exo_mask = ('downsampled_data//' +
                    '//'.join([s_id, x_cam, obj, 'mask', index]))

        obj_pair = [
            [ego_rgb, ego_mask],
            [exo_rgb, exo_mask],
        ]

        scene_dict[s_id].append(obj_pair)

    result = list(scene_dict.values())

    # ---------------------------------------------------------
    # save
    # ---------------------------------------------------------
    with open(save_path, 'w') as f:
        json.dump(result, f)

    print(f'[INFO] saved to {save_path}')


if __name__ == '__main__':
    main()

# python gen_scenes.py --mode val --save_path val_scenes.json
