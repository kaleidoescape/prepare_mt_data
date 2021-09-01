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

from tqdm import tqdm
from pydantic import BaseModel, validator

import utils

#be explicit, so that logging occurs even if this is run as main
logger = utils.setup_logger('resample_parallel')
logger.setLevel(logging.INFO)

SAMPLING_METHODS = [
    'uniform',
    'temperature',
    'sinkhorn',
    'original'
]

class DatasetArg(BaseModel):
    r"""A data set field from the config file."""
    src_lang: str
    tgt_lang: str
    src: str 
    tgt: str 
    weight: Optional[float]=1.0

    @validator('src', 'tgt')
    def valid_file(cls, data):
        if data and not os.path.exists(data): 
            raise FileNotFoundError(f"File not found: {data}")
        #if data and os.stat(data).st_size == 0:
        #    raise OSError(f"File empty: {data}")
        return data

class ConfigArgs(BaseModel):
    r"""The fields from the config file."""
    data: Dict[str, DatasetArg]
    args: Optional[dict]={}


def clean_line(line):
    line = line.replace('\r', '') #windows line ending carraige return
    line = line.replace('\x00', '') #null byte
    line = line.replace('\u200c', '') #zero width non joiner
    line = line.replace('\ufeff', '') #zero width non breaking space
    return line

def original_sampling(
        langs, metadata
    ):
    total = sum([meta['size'] for meta in metadata])
    updated_metadata = []
    for meta in metadata:
        new_meta = meta.copy() #copy to avoid overwriting original dicts
        new_meta['prob'] = meta['size'] / total
        updated_metadata.append(new_meta)

    return updated_metadata

def uniform_sampling_distribution(
        langs, metadata
    ):
    r"""
    Create a uniform distribution over the languages.

    Return an ordered lsit of the sampling distrib for langs and the ordered
    list of langs.
    """
    slangs = sorted(langs)
    distrib = [1 / len(slangs)]*len(slangs)
    return distrib, slangs

def uniform_sampling(
        langs, metadata
    ):
    r"""
    Return an updated metadata dict with the probability assigned to the
    dataset based on its target language. Probability is uniform over langs.
    """
    probs, langs = uniform_sampling_distribution(langs, metadata)

    updated_metadata = [] 
    for meta in metadata:
        src_lang, tgt_lang, sizes = meta['src_lang'], meta['tgt_lang'], meta['size']
        if src_lang not in langs or tgt_lang not in langs:
            logger.warning(f"Dataset {meta['name']} langs {src_lang}2{tgt_lang} do not exist in langs {langs}")
            prob = 0 
        else:
            src_idx, tgt_idx = langs.index(src_lang), langs.index(tgt_lang)
            prob = probs[tgt_idx] #sample based on the target language

        new_meta = meta.copy() #copy to avoid overwriting original dicts
        new_meta['prob'] = prob
        updated_metadata.append(new_meta)

    return updated_metadata

def temperature_sampling_distribution(
        langs, metadata, tmpr=1.0
    ):
    r"""
    Convert dataset sizes into a distribution which takes into account the
    prevalence of the source/target languages in the data.
    
    Return an ordered list of the sampling distrib for langs and the ordered
    list of langs.
    """
    slangs = sorted(langs)
    sizes = [0]*len(slangs)
    for i, meta in enumerate(metadata):
        src_lang, tgt_lang, size = meta['src_lang'], meta['tgt_lang'], meta['size']
        src_idx, tgt_idx = slangs.index(src_lang), slangs.index(tgt_lang)
        sizes[slangs.index(src_lang)] += size
        sizes[slangs.index(tgt_lang)] += size

    sizes = np.asarray(sizes)
    distrib = sizes ** (1 / tmpr)
    distrib = distrib / sum(distrib)
    return distrib, slangs

def temperature_sampling(
        langs, metadata, tmpr=1.0
    ):
    r"""
    Return an updated metadata dict with the probability assigned to the
    dataset based on its target language.
    """
    probs, langs = temperature_sampling_distribution(langs, metadata, tmpr)

    updated_metadata = [] 
    for meta in metadata:
        src_lang, tgt_lang, sizes = meta['src_lang'], meta['tgt_lang'], meta['size']
        if src_lang not in langs or tgt_lang not in langs:
            logger.warning(f"Dataset {meta['name']} langs {src_lang}2{tgt_lang} do not exist in langs {langs}")
            prob = 0 
        else:
            src_idx, tgt_idx = langs.index(src_lang), langs.index(tgt_lang)
            prob = probs[tgt_idx] #sample based on the target language

        new_meta = meta.copy() #copy to avoid overwriting original dicts
        new_meta['prob'] = prob
        updated_metadata.append(new_meta)

    return updated_metadata

def sinkhorn_temperature_sampling_distribution(
        langs, metadata, tmpr=1.0
    ):
    r"""
    Convert dataset sizes into a distribution which takes into account both the
    availability of a particular lang pair together, as well as the
    availability of a particular lang alone across the pairs. We use the
    Sinkhorn-Knopp algorithm to convert a matrix of lang pair counts into 
    a doubly stochastic matrix, which is then converted into the temperature 
    sampled probabilities.

    Return a matrix for the sampling distrib of lang pairs and the ordered
    list of langs that represent the two axes of the matrix.

    Motivation (section 3.4): https://arxiv.org/abs/2010.11125
    Sinkhorn-Knopp paper: http://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf
    Our fork of skp: https://github.com/kaleidoescape/sinkhorn_knopp
    """
    from sinkhorn_knopp import sinkhorn_knopp as skp

    #fill a matrix with language pair counts across datasets
    slangs = sorted(langs)
    A = np.zeros((len(slangs), len(slangs))) #(src, tgt)
    for i, meta in enumerate(metadata):
        src_lang, tgt_lang, size = meta['src_lang'], meta['tgt_lang'], meta['size']
        src_idx, tgt_idx = slangs.index(src_lang), slangs.index(tgt_lang)
        A[src_idx, tgt_idx] += size
    logger.info(f"Data counts for {slangs}:\n{A}")

    #if any row is fully 0, we have to remove the lang from both axes because
    #we need a square matrix with total support to perform sinkhorn-knopp
    #This will cause us to miss a lang that is ever only used as src or only
    #used as tgt, but for multiling models, we typically use both directions
    sorted_langs = slangs.copy() 
    zero_rows = np.where(~A.any(axis=0))[0]
    if zero_rows.size > 0:
        logger.warning(
            f"Ignoring all datasets for langs {[sorted_langs[i] for i in zero_rows]}"
            " because this lang is never used as the src")
    A = np.delete(A, zero_rows, 0)
    A = np.delete(A, zero_rows, 1)
    #also if any col is fully 0, we have to remove the lang from both axes 
    zero_cols = np.where(~A.any(axis=1))[0]
    if zero_cols.size > 0:
        logger.warning(
            f"Ignoring all datasets for langs {[sorted_langs[i] for i in zero_cols]}"
            " because this lang is never used as the tgt")
    A = np.delete(A, zero_cols, 0)
    A = np.delete(A, zero_cols, 1)
    if not A.any():
        raise AttributeError(f"Useable data counts are all 0.")
    #also remove the langs we dropped from the slang list that we'll return
    [slangs.remove(sorted_langs[i]) for i in zero_cols]
    [slangs.remove(sorted_langs[i]) for i in zero_rows]
    if zero_rows.size > 0 or zero_cols.size > 0:
        logger.info(f"Remaining data counts for {slangs}:\n{A}")

    #make matrix doubly stochastic (rows and cols each sum to 1)
    #and convert that into a new probability distrib with temperature
    sk = skp.SinkhornKnopp()
    A_ds = sk.fit(A)
    logger.info(f"Sinkhorn sampled probs for {slangs}:\n{A_ds}")
    probs = A_ds ** (1 / tmpr)
    probs = probs / sum(probs)
    logger.info(f"Sinkhorn temperature (T={tmpr}) sampled probs for {slangs}:\n{probs}")
    return probs, slangs

def sinkhorn_temperature_sampling(
        langs, metadata, tmpr=1.0
    ):
    r"""
    Return an updated metadata dict with a probability assigned to each 
    dataset given its language pairs. Datasets with langs never used for 
    either translating into or translating out of will get a prob of 0.
    """
    probs, langs = sinkhorn_temperature_sampling_distribution(langs, metadata, tmpr)

    updated_metadata = [] 
    for meta in metadata:
        src_lang, tgt_lang, sizes = meta['src_lang'], meta['tgt_lang'], meta['size']
        if src_lang not in langs or tgt_lang not in langs:
            logger.warning(f"Dataset {meta['name']} langs {src_lang}2{tgt_lang} do not exist in sinkhorn langs {langs}")
            prob = 0 
        else:
            src_idx, tgt_idx = langs.index(src_lang), langs.index(tgt_lang)
            prob = probs[src_idx, tgt_idx] #sample based on the language pair

        new_meta = meta.copy() #copy to avoid overwriting original dicts
        new_meta['prob'] = prob
        updated_metadata.append(new_meta)

    return updated_metadata

def main(
        pconfig: str,
        outdir: str,
        n: int,
        sampling_method: Optional[str]='sinkhorn', 
        temperature: Optional[float]=5.0,
        add_tag=False,
        dry_run=False,
    ):
    r"""
    Sample n data points from the data files in the pconfig 'data' 
    using the Sinkhorn-Knopp algorithm to create a doubly stochastic matrix
    for sampling from language pair probabilities. Upsample using temperature
    sampling, so that lower resource pairs are seen more often than they
    otherwise would be according to their low counts. 
    """
    meta = []
    dataset_names = []
    dataset_lengths = {}
    langs = set()
    for k in pconfig['data']:
        src_lang, tgt_lang = pconfig['data'][k]['src_lang'], pconfig['data'][k]['tgt_lang']
        src_data, tgt_data = pconfig['data'][k]['src'], pconfig['data'][k]['tgt']
        langs.add(src_lang)
        langs.add(tgt_lang)

        length = utils.get_file_length(src_data)
        weight = 1.0
        if 'weight' in pconfig['data'][k]:
            weight = pconfig['data'][k]['weight']

        pconfig['data'][k]['size'] = length
        pconfig['data'][k]['name'] = k
        pconfig['data'][k]['weight'] = weight
        meta.append(pconfig['data'][k])
        dataset_names.append(k)
        dataset_lengths[k] = length

    #sample by lang or lang pair, then get the probs for each of the
    #datasets, converted into a distribution over the datasets
    sorted_langs = sorted(langs)
    if sampling_method == 'sinkhorn':
        new_meta = sinkhorn_temperature_sampling(langs, meta, temperature)
    elif sampling_method == 'temperature':
        new_meta = temperature_sampling(langs, meta, temperature)
    elif sampling_method == 'uniform':
        new_meta = uniform_sampling(langs, meta)
    elif sampling_method == 'original':
        new_meta = original_sampling(langs, meta)
    weighted = np.asarray([d['prob']*d['weight'] for d in new_meta])
    dataset_probs = weighted / sum(weighted)
    logger.info(f"Resampled probs:\n{dict(list(zip(dataset_names, dataset_probs)))}")

    if not dry_run: #avoid overwriting files if this isn't a real run 
        dataset_fhs = {
            dataset_names[i]: [
                open(d['src'], 'r', encoding='utf-8'),
                open(d['tgt'], 'r', encoding='utf-8'),
            ]
            for i, d in enumerate(new_meta)
        }

        os.makedirs(outdir, exist_ok=True)

        #add i to name in case of collision when a file is amongst srcs & tgts
        lang_out_fhs = {
            dataset_names[i]: [
                open(os.path.join(outdir, f"{i}.{os.path.basename(d['src'])}.INPUT"), 'w', encoding='utf-8'),
                open(os.path.join(outdir, f"{i}.{os.path.basename(d['tgt'])}.OUTPUT"), 'w', encoding='utf-8'),
            ]
            for i, d in enumerate(new_meta)
        }

    #sample one sentence at a time from the datasets
    final_counts = {k:0 for k in dataset_names}
    read_lines = {k:0 for k in dataset_names}
    c = 0
    pbar = tqdm(total=n)
    while c < n:

        #sample from dataset distribution for a particular dataset to use
        dataset_idx = np.argmax(np.random.multinomial(1, dataset_probs, 1)[0])
        dataset = dataset_names[dataset_idx]

        if dry_run: #don't try to write to files if this isn't a real run
            continue

        #go around the dataset in a circle when upsampling
        read_lines[dataset] += 1
        if read_lines[dataset] == dataset_lengths[dataset]-1:
            dataset_fhs[dataset][0].seek(0)
            dataset_fhs[dataset][1].seek(0)
            read_lines[dataset] = 0

        try:
            src_line = dataset_fhs[dataset][0].readline()
            tgt_line = dataset_fhs[dataset][1].readline()
        except UnicodeDecodeError as e:
            #NOTE: errors could point to a deeper problem with line endings
            logger.warning(f"UnicodeDecodeError from {dataset} (line {read_lines[dataset]}): {line}")
            continue
        else:
            src_line = clean_line(src_line).strip()
            tgt_line = clean_line(tgt_line).strip()
            if add_tag:
                src_line = f"<2{new_meta[dataset_idx]['tgt_lang']}> {src_line}"
            if src_line and tgt_line:
                lang_out_fhs[dataset][0].write(src_line + '\n')
                lang_out_fhs[dataset][1].write(tgt_line + '\n')
                final_counts[dataset] += 1
                pbar.update(1)
                c += 1
    pbar.close()

    #don't forget to close all of the opened files
    if not dry_run:
        [dataset_fhs[k][0].close() for k in dataset_fhs]
        [dataset_fhs[k][1].close() for k in dataset_fhs]
        [lang_out_fhs[k][0].close() for k in lang_out_fhs]
        [lang_out_fhs[k][1].close() for k in lang_out_fhs]

    return final_counts


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
        description="Resample from datasets using sinkhorn temperature sampling over the language pairs.",
        epilog=epilog
    )
    parser.add_argument('--configs', nargs='+', required=True,
        help='path to yaml config file(s) with source/target data (later override earlier)')
    parser.add_argument('--outdir', required=True,
        help="output directory where to save results")
    parser.add_argument('--n', required=True, type=int,
        help="the number of new sentences to sample")
    parser.add_argument('--sampling', required=True, type=str, choices=SAMPLING_METHODS,
        help="the type of sampling method to use")
    parser.add_argument('--tmpr', default=5.0, type=float,
        help="temperature sampling parameter T (only used for temperature sampling and sinkhorn temperature sampling)") 
    parser.add_argument('--tag', default=False, action='store_true',
        help="add tag like <2xx> to the src side with the tgt lang") 
    parser.add_argument('--dry-run', default=False, action='store_true',
        help="do a dry run to make the calculations for the datasets without writing new files") 
    args = parser.parse_args()
    
    args.config = utils.parse_configs(args.configs, args.outdir, parser=ConfigArgs)
    return args

if __name__ == '__main__':
    args = parse_args()
    final_counts = main(
        args.config,
        outdir=args.outdir,
        n=args.n,
        sampling_method=args.sampling,
        temperature=args.tmpr,
        add_tag=args.tag,
        dry_run=args.dry_run,
    )
    logger.info(f"Final counts (total {sum(final_counts.values())}):")
    m = {k.split('_', maxsplit=1)[1]:k for k in final_counts}
    for k in sorted(m):
        logger.info(f"{final_counts[m[k]]} {m[k]}")
