import os
import re
from itertools import permutations

import yaml

import utils

logger = utils.setup_logger('collect_data_files')

def collect_data_files(direc, langs, suffix=None, prefix=None, exclude=None, require=None, ordered=None):
    directions = list(permutations(langs, 2))
    bases = set()
    for root, dirs, files in os.walk(direc):
        for filename in files:
            if prefix and not filename.startswith(prefix):
                continue

            if suffix and not filename.endswith(suffix):
                continue
            elif suffix:
                filename = filename.replace(suffix, '')

            base, ext = os.path.splitext(filename)
            base = os.path.join(root, base)
            bases.add(base)
    
    data = {}
    for direction in directions:
        for base in bases:
            src_lang = direction[0]
            tgt_lang = direction[1]
            src_fp = os.path.join(base + f'.{src_lang}')
            tgt_fp = os.path.join(base + f'.{tgt_lang}')

            if ordered == 'forward':
                if langs.index(src_lang) > langs.index(tgt_lang):
                    continue
            elif ordered == 'backward':
                if langs.index(src_lang) < langs.index(tgt_lang):
                    continue

            if suffix:
                src_fp = src_fp + suffix
                tgt_fp = tgt_fp + suffix

            #do these checks after we've concluded that we'd otherwise keep it
            #and have constructed the accurate file path
            exclude_ok = True
            if exclude:
                for item in exclude:
                    if item in src_fp or item in tgt_fp:
                        exclude_ok = False
                        break
            require_ok = True
            if require:
                require_ok = False
                for item in require:
                    if item in src_fp or item in tgt_fp:
                        require_ok = True
                        break
            if not (exclude_ok and require_ok):
                continue

            if os.path.exists(src_fp) and os.path.exists(tgt_fp):
                #try to simplify the name for reader's ease
                dataset_name, ext = os.path.splitext(base)
                name = os.path.relpath(dataset_name, direc)
                name = name.replace('/', '-')
                k = f"{src_lang}2{tgt_lang}_{name}"
                #logger.info(f"Adding: {src_fp} {k}")

                data[k] = {
                    'src_lang': src_lang,
                    'tgt_lang': tgt_lang,
                    'src': src_fp,
                    'tgt': tgt_fp
                }
    return data

def main(direc, outfp, langs, suffix=None, prefix=None, exclude=None, require=None, ordered=None):
    r"""
    Args:
        direc: directory to search through for files
        outfp: filename to save the yaml output
        langs: list of language suffixes to search for
        suffix: all files must have this immediately after the lang suffix
        prefix: all files must start with this 
        exclude: no file may have any one of these strings
        ordered: [None, forward, backward]: the list of langs is ordered 
            and we do not add the opposite direction to the config file
            (None means we add both directions into the config file)

    """
    data = {'data': 
        collect_data_files(
            direc, 
            langs, 
            suffix=suffix, 
            prefix=prefix, 
            exclude=exclude,
            require=require, 
            ordered=ordered,
        )
    }
    with open(outfp, 'w', encoding='utf-8') as fh:
        yaml.dump(data, fh)

if __name__ == '__main__':
    import fire
    fire.Fire(main)


        
