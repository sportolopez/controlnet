import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass

        return res

def confirm(inp):
    confirm = input(f'{inp} Continue? [Y/n]: ').lower().strip()
    if confirm in ['no','n'] or confirm not in ['yes', 'y', '']:
        return False
    return True

def return_found_keys(keys: list[str], d: dict):
    for key in keys:
        if key in d:
            yield key, d[key]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, default=None, metavar='PATH', help="Root directory where LoRAs models are stored.")
    parser.add_argument("--repeat-threshold", type=int, required=False, default=100, help="Threshold for image repeats. Values >=100 are likely bad.")
    parser.add_argument("--repeat-image-ratio-threshold", type=float, required=False, default=1.01, help="Repeats / image count. Values >=1 are possibly bad.")
    parser.add_argument("--dim-threshold", type=int, required=False, default=129, help="dim / rank. Values >128 are likely bad.")
    parser.add_argument("--tag-count-threshold", type=int, required=False, default=3, help="Count of tags in a dataset. Smaller values are likely bad.")
    parser.add_argument("--tag-length-threshold", type=int, required=False, default=100, help="Max length of a tag in a dataset. Larger values are likely bad.")
    parser.add_argument("--out", type=str, required=False, default='lora-scan-report.txt', metavar='PATH', help="Output location of report. Defaults to working directory.")
    args = parser.parse_args()

    src = Path(args.src)
    to_inspect: list[dict] = []
    ss_keys_dataset = ['ss_reg_dataset_dirs', 'ss_dataset_dirs']

    print('Scanning...')
    files = list(glob(str(src.joinpath('**/*.safetensors')), recursive=True))
    if not len(files) > 0:
        print('No LoRAs found (files with a .safetensors extension), exiting')
        exit(1)
    for f in tqdm(files):
        meta = read_metadata_from_safetensors(f)
        if bool(meta):
            network_data = {}

            try:
                repeats = []
                repeat_image_ratio = []
                for _key, res in return_found_keys(ss_keys_dataset, meta):
                    res: dict
                    if not bool(res):
                        break
                    repeats += [int(x['n_repeats']) for x in res.values()]
                    repeat_image_ratio += [int(x['n_repeats']) / int(x['img_count']) for x in res.values() if int(x['img_count']) != 0]

                if 'ss_tag_frequency' in meta:
                    #print(next(iter(meta['ss_tag_frequency'])).keys())
                    frequencies = [1 if len(val.keys()) == 0 else len(val.keys()) for key, val in meta['ss_tag_frequency'].items() if key not in ['reg_data'] and not key.endswith('.json')]
                    tags = list(np.concatenate([list(val.keys()) for _key, val in meta['ss_tag_frequency'].items()]))
                    nombre_archivo = os.path.basename(f)

                    print(nombre_archivo + ": " +tags[0])
                    if len(tags) > 0:
                        tags.sort(key=len, reverse=True)
                        max_tag_length = max([len(x) for x in tags])
                        if max_tag_length >= args.tag_length_threshold:
                            network_data['tag_length_threshold'] = max_tag_length

                    if len(frequencies) > 0:
                        if max(frequencies) <= args.tag_count_threshold:
                            network_data['tag_count_threshold'] = max(frequencies)
                        if all(x <= args.tag_count_threshold for x in frequencies):
                            network_data['tag_count_threshold_all'] = True

                    for key, _vals in meta['ss_tag_frequency'].items():
                        if key.endswith('.json'):
                            network_data['invalid_tag_frequency_data_key'] = key
                            break

                if any(x >= args.repeat_threshold for x in repeats):
                    network_data['repeat_threshold'] = max(repeats)

                if any(x >= args.repeat_image_ratio_threshold for x in repeat_image_ratio):
                    network_data['repeat_image_ratio_threshold'] = f'{max(repeat_image_ratio):.2f}'

                try:
                    if int(meta['ss_network_dim']) >= args.dim_threshold:
                        network_data['dim_threshold'] = int(meta['ss_network_dim'])
                except ValueError:
                    pass
            except KeyError:
                pass

            if len(network_data.keys()) > 0:
                if len(network_data.keys()) > 1:
                    network_data['meta_multiple'] = True
                network_data['meta_path'] = str(f)
                to_inspect.append(network_data)

    unique_keys = []
    for f in to_inspect:
        for key in f.keys():
            key: str
            if key.startswith('meta_'):
                continue
            if key not in unique_keys:
                unique_keys.append(key)

    with open(args.out, mode='w+', encoding='utf8') as fd:
        for key in unique_keys:
            fd.write(f'{key.upper()}\n')
            for network in to_inspect:
                if key in network.keys():
                    fd.write(f'{network[key]} :: {network["meta_path"]}\n')
            fd.write('\n\n')
        fd.write(f'FINAL SCORE (total LoRAs found): {len(to_inspect)} ({len(to_inspect) / len(files) * 100:.1f}%)')

    print('Done!')