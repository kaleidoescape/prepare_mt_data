import json
import logging
import os
import shutil
import subprocess
from typing import *
from typing import IO # the * misses this one

import utils

logger = logging.getLogger('multimeta')
logger.setLevel(logging.INFO)

def prepare_metdata(fps: list, langs: list, outdir: str):
    r"""
    Return a tuple of dicts (lang_to_idx, line_offsets), where:
        lang_to_idx: lang to list of lines that have translations in the lang
        line_offests: file path to byte offsets for each line in the file
    """
    length = utils.get_file_length(fps[0])
    for fp in fps[1:]: 
        assert utils.get_file_length(fp) == length, f'files differ in length: {fps}'
    
    fhs = [open(fp, 'r', encoding='utf-8') for fp in fps]
    lang_to_idx = {lang: [] for lang in langs}
    idx_to_lang = {}
    line_offsets = {fp: {} for fp in fps}
    for ln in range(length):
        idx_to_lang[ln] = []
        for i, lang in enumerate(langs):
            #get the starting position of this line
            line_offsets[fps[i]][ln] = fhs[i].tell()
            #advance this line
            line = fhs[i].readline()
            if line.strip():
                lang_to_idx[lang].append(ln)
                idx_to_lang[ln].append(lang)

    return lang_to_idx, idx_to_lang, line_offsets

def main(data: dict, outdir: str):
    for dataset_name in data:
        print(f"Creating metadata for {dataset_name}")
        fps = [data[dataset_name]['src'], data[dataset_name]['tgt']]
        langs = [data[dataset_name]['src_lang'], data[dataset_name]['tgt_lang']]
        metadata = prepare_metdata(fps, langs, outdir)

        multiway_meta_fp = os.path.join(outdir, f'{dataset_name}.meta')
        with open(multiway_meta_fp, 'w', encoding='utf-8') as outfile:
            metadata = {
                'files': {langs[i]: fps[i] for i in range(len(fps))},
                'lang_to_idx': metadata[0], #line numbers for each lang
                'idx_to_lang': metadata[1], #each lang that is on a line
                'offsets': metadata[2], #line offsets for lines in the files
            }
            utils.write_json_line(metadata, outfile)

if __name__ == '__main__':
    import argparse
    import collections.abc
    import yaml
    from pydantic import BaseModel, validator

    def parse_args():
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
            description="Create metadata for multilingual dataset.",
        )
        parser.add_argument('--configs', nargs='+', required=True,
            help='path to yaml config file(s) with source/target data (later override earlier)')
        parser.add_argument('--outdir', required=True,
            help="output directory where to save the evaluation results directory")
        args = parser.parse_args()
        
        args.config = utils.parse_configs(args.configs, args.outdir) 
        return args
    
    args = parse_args()
    main(args.config['data'], args.outdir)
