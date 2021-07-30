import os
import re
from itertools import permutations
from typing import *
from typing import IO #the star doesn't pick this one up

import yaml

import utils

logger = utils.setup_logger('collect_data_files')

def collect_data_files(
        direc: str, 
        langs: list, 
        suffix: Optional[str]=None, 
        prefix: Optional[str]=None, 
        exclude: Optional[list]=None, 
        require: Optional[list]=None, 
        ordered: Optional[str]=None, #{None, forward, backward}
        unique: Optional[bool]=True,
        mono: Optional[bool]=True,
        skip: Optional[list]=None, #e.g. ['xx2yy', 'yy2zz']
    ):
    assert ordered in [None, False, 'forward', 'backward']

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
    seen = set()
    for direction in directions:
        if skip and '2'.join(direction) in skip:
            continue
        for base in bases:
            src_lang = direction[0]
            tgt_lang = direction[1]

            src_fp = base + f'.{src_lang}'
            tgt_fp = base + f'.{tgt_lang}'
            
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
            
            name = os.path.relpath(base, direc)
            name = name.replace('/', '-')

            if mono:
                if os.path.exists(src_fp):
                    k = f"{src_lang}_mono_{name}"
                    src_name = os.path.basename(src_fp)
                    if src_name in seen:
                        continue
                    seen.add(src_name)
                    data[k] = {
                        'src_lang': src_lang,
                        'tgt_lang': '',
                        'src': src_fp,
                        'tgt': '' 
                    }
                if os.path.exists(tgt_fp):
                    k = f"{tgt_lang}_mono_{name}"
                    tgt_name = os.path.basename(tgt_fp)
                    if tgt_name in seen:
                        continue
                    seen.add(tgt_name)
                    data[k] = {
                        'src_lang': tgt_lang,
                        'tgt_lang': '',
                        'src': tgt_fp,
                        'tgt': '' 
                    }

            elif os.path.exists(src_fp) and os.path.exists(tgt_fp):
                k = f"{src_lang}2{tgt_lang}_{name}"

                src_name = os.path.basename(src_fp)
                tgt_name = os.path.basename(tgt_fp)
                if unique and src_name in seen and tgt_name in seen:
                    continue
                seen.add(src_name)
                seen.add(tgt_name)

                data[k] = {
                    'src_lang': src_lang,
                    'tgt_lang': tgt_lang,
                    'src': src_fp,
                    'tgt': tgt_fp
                }
    if not data:
        logger.warning("No files found, config is empty!")
    return data

def main(
        direc, 
        outfp, 
        langs, 
        suffix=None, 
        prefix=None, 
        exclude=None, 
        require=None, 
        ordered=None, 
        unique=True,
        mono=False,
        skip=None,
    ):
    r"""
    Args:
        direc: directory to search through for files
        outfp: filename to save the yaml output
        langs: list of language suffixes to search for
        suffix: all files must have this immediately after the lang suffix
        prefix: all files must start with this 
        exclude: no file may have any one of these strings
        ordered: {None, forward, backward}: the list of langs is ordered 
            and we do not add the opposite direction to the config file
            (None means we add both directions into the config file)
        unique: don't use the same base filename twice, to prevent creating
            mirrored backwards directions when the file has already been used
        mono: search for monolingual data (also ending in lang suffix)
        skip: list of language directions to skip in the form ['xx2yy']

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
            unique=unique,
            mono=mono,
            skip=skip,
        )
    }
    with open(outfp, 'w', encoding='utf-8') as fh:
        yaml.dump(data, fh)

if __name__ == '__main__':
    import fire
    fire.Fire(main)


        
