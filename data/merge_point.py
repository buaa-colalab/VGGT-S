import argparse
import json

# =========================================================
# Merge
# =========================================================


def merge_json_by_key(
    json_path_1,
    json_path_2,
):

    with open(json_path_1, 'r', encoding='utf-8') as fp:
        data_1 = json.load(fp)

    with open(json_path_2, 'r', encoding='utf-8') as fp:
        data_2 = json.load(fp)

    merged = {}

    all_keys = set(data_1.keys()) | set(data_2.keys())

    for key in all_keys:

        merged[key] = {}

        if key in data_1:
            merged[key].update(data_1[key])

        if key in data_2:
            merged[key].update(data_2[key])

    print(f'[NOTE] merged total keys: {len(merged)}')

    return merged


# =========================================================
# Dict -> List
# =========================================================


def convert_dict_to_list(data):

    results = []

    for key, value in data.items():

        item = {
            'scene_info': key,
        }

        item.update(value)

        results.append(item)

    return results


# =========================================================
# Main
# =========================================================


def main(
    json_path_1,
    json_path_2,
    save_path,
):

    # =====================================================
    # Merge
    # =====================================================

    merged_data = merge_json_by_key(
        json_path_1=json_path_1,
        json_path_2=json_path_2,
    )

    # =====================================================
    # Convert
    # =====================================================

    results = convert_dict_to_list(merged_data)

    # =====================================================
    # Save
    # =====================================================

    with open(save_path, 'w', encoding='utf-8') as fp:

        json.dump(results, fp, indent=4)

    print(f'[NOTE] saved to: {save_path}')
    print(f'[NOTE] total samples: {len(results)}')


# =========================================================
# Entry
# =========================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'val'])

    parser.add_argument(
        '--json_path_1',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--json_path_2',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # =====================================================
    # Default Paths
    # =====================================================

    json_path_1 = (args.json_path_1 if args.json_path_1 is not None else
                   f'{args.mode}_obj_wp_g2x.json')

    json_path_2 = (args.json_path_2 if args.json_path_2 is not None else
                   f'{args.mode}_obj_wp_x2g.json')

    save_path = (args.save_path
                 if args.save_path is not None else f'{args.mode}_obj_wp.json')

    # =====================================================
    # Run
    # =====================================================

    main(
        json_path_1=json_path_1,
        json_path_2=json_path_2,
        save_path=save_path,
    )
