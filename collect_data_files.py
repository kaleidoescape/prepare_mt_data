import os
from itertools import permutations

import yaml

def collect_data_files(direc, langs, suffix=None, prefix=None):
    directions = list(permutations(langs, 2))
    bases = set()
    for root, dirs, files in os.walk(direc):
        for filename in files:
            base, ext = os.path.splitext(filename)
            if prefix and not filename.startswith(prefix):
                continue
            if suffix and not filename.endswith(suffix):
                continue
            bases.add(os.path.join(root, base))

    data = {}
    for direction in directions:
        for base in bases:
            src_lang = direction[0]
            tgt_lang = direction[1]
            src_fp = os.path.join(base + f'.{src_lang}')
            tgt_fp = os.path.join(base + f'.{tgt_lang}')
            if suffix:
                src_fp = src_fp + suffix
                tgt_fp = tgt_fp + suffix
            if os.path.exists(src_fp) and os.path.exists(tgt_fp):
                #try to simplify the name for reader's ease
                dataset_name, ext = os.path.splitext(base)
                name = os.path.relpath(dataset_name, direc)
                name = name.replace('/', '-')
                data[f'{src_lang}2{tgt_lang}_{name}'] = {
                    'src_lang': src_lang,
                    'tgt_lang': tgt_lang,
                    'src': src_fp,
                    'tgt': tgt_fp
                }
    return data

def main(direc, outfp, langs, suffix=None, prefix=None):
    data = {'data': collect_data_files(direc, langs, suffix=suffix)}
    with open(outfp, 'w', encoding='utf-8') as fh:
        yaml.dump(data, fh)

if __name__ == '__main__':
    import fire
    fire.Fire(main)


        
