import argparse
import json
import logging
import os
import shutil
import subprocess
import textwrap
from typing import *

import yaml
import numpy as np

import utils

#be explicit, so that logging occurs even if this is run as main
logger = utils.setup_logger('resample_to_mono')
logger.setLevel(logging.INFO)

def temperature_sampling(sizes, temperature=5):
    sizes = np.asarray(sizes)
    distrib = sizes ** (1 / temperature)
    return distrib / sum(distrib)

def original_distrib_sampling(sizes):
    sizes = np.asarray(sizes)
    return sizes / sum(sizes)

def uniform_sampling(sizes):
    return np.asarray([1/len(sizes) for i in sizes])

def main(
        pconfig: str,
        outdir: str,
        n: int,
        sampling_method: Optional[str]='uniform', 
        temperature: Optional[float]=5,
        dataset_sampling_method: Optional[str]='uniform', 
        dataset_temperature: Optional[float]=5,
    ):
    r"""
    Resample multilingual or monolingual training data into new 
    MONOLINGUAL files using a particular sampling distribution. 

    NOTE: We sample from datasets line-by-line, and do not pre-shuffle
    the dataset, so if this is necessary, please pre-shuffle it first.

    Args:
        config: config file with test sets
        outdir: output directory to put intermediate files into
        n: number of sentences to sample into the new output files
        sampling_method: {temperature,uniform,orig}
        temperature: T parameter for sampling from a lang (only used when
            samping_method==temperature)
        dataset_sampling_method: {temperature,uniform,orig}
        dataset_temperature: T parameter for sampling from a lang-specific 
            dataset (only used when dataset_sampling_method==temperature)

    Example config:
        { data: { 
                my_parallel_dataset: {
                    src_lang: xx,
                    tgt_lang: yy,
                    src: source_file
                    tgt: target_file 
                },
                my_monolingual_dataset: {
                    src_lang: xx,
                    tgt_lang: '',
                    src: source_file
                    tgt: ''
                },
            }
        }
    """
    os.makedirs(outdir, exist_ok=True)

    langs = {}
    for k in pconfig['data']:
        src_lang, tgt_lang = pconfig['data'][k]['src_lang'], pconfig['data'][k]['tgt_lang']
        src_data, tgt_data = pconfig['data'][k]['src'], pconfig['data'][k]['tgt']

        if src_lang and src_data and src_lang not in langs:
            langs[src_lang] = [src_data]
        elif src_lang and src_data:
            langs[src_lang].append(src_data)
        if tgt_lang and tgt_data and tgt_lang not in langs:
            langs[tgt_lang] = [tgt_data]
        elif tgt_lang and tgt_data:
            langs[tgt_lang].append(tgt_data)

    sorted_langs = sorted(langs.keys())

    #prepare metadata
    lengths = {}
    dataset_lengths = {}
    dataset_distribs = {}
    dataset_fhs = {}
    for lang in sorted_langs:
        lengths[lang] = [utils.get_file_length(fp) for fp in langs[lang]]
        for i, fp in enumerate(langs[lang]):
            dataset_lengths[fp] = lengths[lang][i]

        #create the sampling distribution for the datasets of this lang
        if dataset_sampling_method == 'temperature':
            dataset_distribs[lang] = temperature_sampling(lengths[lang], dataset_temperature)
        elif dataset_sampling_method == 'uniform':
            dataset_distribs[lang] = uniform_sampling(lengths[lang])
        elif dataset_sampling_method == 'orig':
            dataset_distribs[lang] = original_distrib_sampling(lengths[lang])

        for fp in langs[lang]:
            dataset_fhs[fp] = open(fp, 'r', encoding='utf-8')
        logger.info(f"Available datapoints for '{lang}' across {len(lengths[lang])} datasets: {sum(lengths[lang])}")

    lang_lengths = [sum(lengths[l]) for l in sorted_langs]

    #create the sampling distribution over the languages
    if sampling_method == 'temperature':
        distrib = temperature_sampling(lang_lengths, temperature)
    elif sampling_method == 'uniform':
        distrib = uniform_sampling(lang_lengths)
    elif sampling_method == 'orig':
        distrib = original_distrib_sampling(lang_lengths)

    total_datapoints = sum(lang_lengths)
    logger.info(f"Sampling {n} samples from total length {total_datapoints}")
    logger.info(f"Resampled distribution for {sorted_langs}: {distrib}")
    sampled_lang_idxs = np.argmax(np.random.multinomial(1, distrib, n), axis=1)
    
    prefix = os.path.join(outdir, 'resampled')
    lang_out_fps = [prefix + f'.{l}' for l in sorted_langs]
    lang_out_fhs = [open(fp, 'w', encoding='utf-8') for fp in lang_out_fps]

    read_lines = {k:0 for k in dataset_fhs}
    for i, lang_idx in enumerate(sampled_lang_idxs):
        lang = sorted_langs[lang_idx]

        #also temperature sample from datasets for this language
        dataset_idx = np.argmax(np.random.multinomial(1, dataset_distribs[lang], 1)[0])
        dataset = langs[lang][dataset_idx]

        #go around the dataset in a circle when upsampling
        read_lines[dataset] += 1
        if read_lines[dataset] == dataset_lengths[dataset]:
            dataset_fhs[dataset].seek(0)
            read_lines[dataset] = 0

        try:
            line = dataset_fhs[dataset].readline()
        except UnicodeDecodeError as e:
            logger.warning(f"UnicodeDecodeError from {dataset} (line {read_lines[dataset]}): {line}")
            continue
        lang_out_fhs[lang_idx].write(line)

    [fh.close() for fh in lang_out_fhs]
    [dataset_fhs[k].close() for k in dataset_fhs]

    return outdir


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
        description="Resample from datasets using temperature sampling over the languages and the datasets for each language.",
        epilog=epilog
    )
    parser.add_argument('--configs', nargs='+', required=True,
        help='path to yaml config file(s) with source/target data (later override earlier)')
    parser.add_argument('--outdir', required=True,
        help="output directory where to save results")
    parser.add_argument('--n', required=True, type=int,
        help="the number of new sentences to sample")
    parser.add_argument('--sampling-method', default='uniform', choices=['temperature', 'uniform', 'orig'],
        help="the sampling method to use for sampling from the language distribution")
    parser.add_argument('--temperature', default=5.0, type=float,
        help="only applicable in case sampling-method==temperature; temperature sampling parameter T for sampling a language; new distribution will be (sizes ** (1/T) / sum(sizes ** (1/T))")
    parser.add_argument('--dataset-sampling-method', default='orig', choices=['temperature', 'uniform', 'orig'],
        help="the sampling method to use for sampling from each dataset within a language")
    parser.add_argument('--dataset_temperature', default=5.0, type=float,
        help="only applicable in case dataset-sampling-method==temperature; temperature sampling parameter T for sampling a datasets under a given language")
    args, rest = parser.parse_known_args()
    args.rest = rest 
    
    args.config = utils.parse_configs(args.configs, args.outdir)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(
        args.config,
        outdir=args.outdir,
        n=args.n,
        sampling_method=args.sampling_method,
        temperature=args.temperature,
        dataset_sampling_method=args.dataset_sampling_method,
        dataset_temperature=args.dataset_temperature,
    )
