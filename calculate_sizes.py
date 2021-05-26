import argparse
import logging
import os
import shutil
import textwrap

import numpy as np

import utils

logger = utils.setup_logger('calculate_sizes')
logger.setLevel(logging.INFO)

def calculate_sizes(data):
    r"""
    Return a tuple of the square matrix of parallel data sizes of 
    shape (src, tgt) and the ordered list of langs for the axes.
    """
    langs = set()
    for k in data:
        src_lang, tgt_lang = data[k]['src_lang'], data[k]['tgt_lang']
        langs.add(src_lang); langs.add(tgt_lang)
    langs = sorted(langs)

    lengths = {} 
    A = np.zeros( (len(langs), len(langs)), dtype=int ) #(src, tgt)
    for k in data:
        src_lang, tgt_lang = data[k]['src_lang'], data[k]['tgt_lang']
        src_data, tgt_data = data[k]['src'], data[k]['tgt']
        if src_data not in lengths or tgt_data not in lengths:
            length = utils.get_file_length(src_data)
            #save time counting by assuming src_length==tgt_length
            lengths[src_data] = lengths[tgt_data] = length
        else: #don't waste time recounting reverse-direction datasets
            length = lengths[src_data]
        src_idx, tgt_idx = langs.index(src_lang), langs.index(tgt_lang)
        A[src_idx, tgt_idx] += length

    return A, langs

def main(pconfig, csv_file):
    r"""Create a csv file of parallel data counts."""
    A, langs = calculate_sizes(pconfig['data'])
    np.savetxt(csv_file, A, delimiter=",", header=",".join(langs), fmt='%f')
    

def parse_args():
    epilog = textwrap.dedent(f"""
    Example config.yml:
    -------------------
    data:
      my_parallel_dataset:
        src_lang: xx
        tgt_lang: yy
        src: src_file_path
        tgt: tgt_file_path
    """)
    
    parser = argparse.ArgumentParser(
        formatter_class=utils.ArgParseHelpFormatter,
        description="Get a matrix of data counts by language pair for parallel data, and save it to a csv file",
        epilog=epilog
    )
    parser.add_argument('--configs', nargs='+', required=True,
        help='path to yaml config file(s) with source/target data (later override earlier)')
    parser.add_argument('--csv', required=True,
        help="csv file to save counts to")
    args = parser.parse_args()
    
    args.config = utils.parse_configs(args.configs)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args.config, args.csv)




