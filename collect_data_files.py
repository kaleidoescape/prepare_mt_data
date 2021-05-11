import os
import re
from itertools import permutations

import yaml

def collect_data_files(direc, langs, suffix=None, prefix=None, ordered=None):
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

            if ordered == 'forward':
                src_search = f"\W{src_lang}\W{tgt_lang}\W.*\.{src_lang}"
                tgt_search = f"\W{src_lang}\W{tgt_lang}\W.*\.{tgt_lang}"
                src_matches = re.search(src_search, src_fp)
                tgt_matches = re.search(tgt_search, tgt_fp)
                if not src_matches or not tgt_matches:
                    continue
            elif ordered == 'backward':
                src_search = f"\W{src_lang}\W{tgt_lang}\W.*\.{tgt_lang}"
                tgt_search = f"\W{src_lang}\W{tgt_lang}\W.*\.{src_lang}"
                src_matches = re.search(src_search, src_fp)
                tgt_matches = re.search(tgt_search, tgt_fp)
                if not src_matches or not tgt_matches:
                    continue

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

def main(direc, outfp, langs, suffix=None, prefix=None, ordered=False):
    r"""
    Args:
        direc: directory to search through for files
        outfp: filename to save the yaml output
        langs: list of language suffixes to search for
        suffix: all files must have this immediately after the lang suffix
        prefix: all files must start with this 
        ordered: [None, forward, backward]: the list of langs is ordered so 
            files must use a specific naming scheme; for 'forward', src must
            come first, such as [src-tgt.src, src-tgt.tgt]
            (i.e. [tgt-src.src, tgt-src.tgt] will NOT be accepted);
            for 'backward' the opposite is true; 
            None means either is accepted

    """
    data = {'data': collect_data_files(direc, langs, suffix=suffix, prefix=prefix, ordered=ordered)}
    with open(outfp, 'w', encoding='utf-8') as fh:
        yaml.dump(data, fh)

if __name__ == '__main__':
    import fire
    fire.Fire(main)


        
