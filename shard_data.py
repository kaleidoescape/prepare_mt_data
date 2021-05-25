import argparse
import logging
import os
import shutil
import textwrap

import numpy as np

import utils

#be explicit, so that logging occurs even if this is run as main
logger = utils.setup_logger('shard_data')
logger.setLevel(logging.INFO)

def shard_data_n_shards(fp, outdir, n_shards=1, length=None):
    r"""
    Split fp into n_shards folders evenly, except for the last folder,
    which will have the remainder lines. Return the number of shards created,
    which may be 1 less than the requested n_shards, in case there were no
    remainder lines.
    """
    if length is None:
        length = utils.get_file_length(fp)
    if n_shards == 1:
        lines_per_shard = length
    lines_per_shard = length // (n_shards - 1)
    if lines_per_shard == 0:
        lines_per_shard = length
    extras = length % (n_shards - 1)

    name = os.path.basename(fp)
    logger.info(f"{name}: shards 1 to {n_shards-1}: {lines_per_shard} lines, shard {n_shards}: {extras} lines")

    [os.makedirs(os.path.join(outdir, f"{i:03d}"), exist_ok=True) for i in range(1, n_shards)]
    shard_fps = [os.path.join(outdir, f"{i:03d}", name) for i in range(1, n_shards)]
    if extras:
        os.makedirs(os.path.join(outdir, f"{len(shard_fps)+1:03d}"), exist_ok=True)
        shard_fps.append(os.path.join(outdir, f"{len(shard_fps)+1:03d}", name)) 
    shard_fhs = [open(fp, 'wb') for fp in shard_fps]
    
    with open(fp, 'rb') as infh:
        for shard in range(1, n_shards+1): 
            if shard != n_shards:
                for i in range(lines_per_shard):
                    line = infh.readline()
                    shard_fhs[shard-1].write(line)
            elif extras:
                for i in range(extras):
                    line = infh.readline()
                    shard_fhs[shard-1].write(line)
            else: 
                n_shards = n_shards - 1
    return n_shards

def shard_data_n_lines(fp, outdir, n_lines=100000, length=None):
    r"""
    Split fp into shards, where each shard has n_lines
    (except last, small shard, which has the remainder).
    """
    if length is None:
        length = utils.get_file_length(fp)
    n_shards = (length // n_lines) + 1
    n_shards = shard_data_n_shards(fp, outdir, n_shards=n_shards, length=length)
    return n_shards

def main(pconfig, outdir, min_n_shards=1, replicate=True):
    r"""
    Split data into shard folders in the outdir. The smallest dataset will be
    split into min_n_shards. Bigger datasets will be split so each of their 
    shards has at most the number of lines of the small dataset / min_n_shards.
    
    If replicate=True, smaller datasets will be replicated into the shards that
    they're missing from (their shards will be looped over for replication, e.g.
    shard 0 gets copied into the first shard the small data is missing from,
    shard 1 into the next shard it's missing from, etc.).
    """
    line_counts = {}
    for k in pconfig['data']:
        src_lang, tgt_lang = pconfig['data'][k]['src_lang'], pconfig['data'][k]['tgt_lang']
        src_data, tgt_data = pconfig['data'][k]['src'], pconfig['data'][k]['tgt']
        length = utils.get_file_length(src_data)
        if src_data and os.path.exists(src_data):
            line_counts[src_data] = length
        if tgt_data and os.path.exists(tgt_data) and tgt_data not in line_counts:
            line_counts[tgt_data] = length

    min_size = np.min(list(line_counts.values())) // min_n_shards

    sharded = {}
    for fp in list(line_counts.keys()): 
        n = shard_data_n_lines(fp, outdir, min_size, line_counts[fp])
        sharded[fp] = n

    max_shard_id = np.max(list(sharded.values()))
    if replicate:
        for fp in sharded:
            name = os.path.basename(fp)
            avail = current = sharded[fp]
            shifting = 1
            while current < max_shard_id:
                src = os.path.join(outdir, f"{shifting:03d}", name)
                dst = os.path.join(outdir, f"{current+1:03d}", name)
                shutil.copy(src, dst)
            
                shifting += 1
                if shifting == avail:
                    shifting = 1
                current += 1
                

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
      my_monolingual_dataset:
        src_lang: xx
        tgt_lang: ''
        src: src_file_path
        tgt: ''
    """)
    
    parser = argparse.ArgumentParser(
        formatter_class=utils.ArgParseHelpFormatter,
        description="Split text data (either parallel or monolingual data) into n_shards pieces inside subfolders in the outdir.",
        epilog=epilog
    )
    parser.add_argument('--configs', nargs='+', required=True,
        help='path to yaml config file(s) with source/target data (later override earlier)')
    parser.add_argument('--outdir', required=True,
        help="output directory where to save results")
    parser.add_argument('--min-n-shards', default=1, type=int,
        help="the minimum number of shards to create (smaller data may go into less shards)")
    parser.add_argument('--no-replicate', default=False, action='store_true',
        help="do not replicate small datasets into other higher-numbered shards")
    args, rest = parser.parse_known_args()
    args.rest = rest 
    
    args.config = utils.parse_configs(args.configs, args.outdir)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args.config, args.outdir, args.min_n_shards)




