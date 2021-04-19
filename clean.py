import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
from functools import partial
from multiprocessing import cpu_count
from typing import *

import fasttext_langid
import retagger
from parallely import pll_single, pll_multi

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CPU_COUNT = max(1, math.floor(cpu_count() / 2) - 1)

logger = logging.getLogger('clean')
logger.setLevel(logging.INFO)

def make_tsv(src_fp, tgt_fp, tsv):
    cmd = f"paste {src_fp} {tgt_fp} > {tsv}"
    logger.info('RUNNING: ' + cmd)
    subprocess.call(cmd, shell=True)
    return tsv

def unmake_tsv(tsv, src_fp, tgt_fp):
    cmd = f"cat {tsv} | cut -f1 > {src_fp}"
    logger.info('RUNNING: ' + cmd)
    subprocess.call(cmd, shell=True)
    cmd = f"cat {tsv} | cut -f2 > {tgt_fp}"
    logger.info('RUNNING: ' + cmd)
    subprocess.call(cmd, shell=True)
    return src_fp, tgt_fp

def _charfix(infp, outfp):
    chars = u"\u200c\x00".encode("utf8") #encode them first 
    cmd = f'cat {infp} | sed -e "s/\r//g" | sed -e "s/[{chars}]//g" > {outfp}'
    logger.info('RUNNING: ' + cmd)
    subprocess.call(cmd, shell=True)
    return outfp

def charfix(src: str, tgt: str, output_dir: str, src_lang: str, tgt_lang: str, **kwargs) -> dict:
    src_outfp = os.path.join(output_dir, os.path.basename(src) + '.charfix')
    tgt_outfp = os.path.join(output_dir, os.path.basename(tgt) + '.charfix')

    #avoid overwriting/redoing work
    if os.path.exists(src_outfp) and os.path.exists(tgt_outfp):
        logger.info(f"Skipping; files already exist: {src_outfp} {tgt_outfp}")
        return {'src': src_outfp, 'tgt': tgt_outfp}

    src_outfp = _charfix(src, src_outfp)
    tgt_outfp = _charfix(tgt, tgt_outfp)
    return {'src': src_outfp, 'tgt': tgt_outfp}

def _bifix(tsv, output_dir, src_lang, tgt_lang):
    output = os.path.join(output_dir, os.path.basename(tsv) +".bifixed")
    cmd = f"python {ROOT_DIR}/bifixclean/bifixer.py --scol 1 --tcol 2"\
            f" {tsv} {output} {src_lang} {tgt_lang} && exit"
    logger.info('RUNNING: ' + cmd)
    subprocess.call(cmd, shell=True)
    return output

def bifix(src: str, tgt: str, output_dir: str, src_lang: str, tgt_lang: str, **kwargs) -> dict:
    src_outfp = os.path.join(output_dir, os.path.basename(src)) + '.bifixed'
    tgt_outfp = os.path.join(output_dir, os.path.basename(tgt)) + '.bifixed'

    #avoid overwriting/redoing work
    if os.path.exists(src_outfp) and os.path.exists(tgt_outfp):
        logger.info(f"Skipping; files already exist: {src_outfp} {tgt_outfp}")
        return {'src': src_outfp, 'tgt': tgt_outfp}

    #setup
    os.makedirs(output_dir, exist_ok=True)
    tsv = os.path.join(output_dir, os.path.basename(src)) + '.tsv'
    make_tsv(src, tgt, tsv)

    #parallel process bifixer, because it's not parallelized
    part = partial(
        _bifix, 
        output_dir=output_dir, 
        src_lang=src_lang, 
        tgt_lang=tgt_lang, 
    )
    bif_tsv = pll_single(
        tsv, 
        part, 
        n_jobs=CPU_COUNT,
        outdir=output_dir, 
        output_name=os.path.basename(tsv) + '.bifixed'
    ) 
    unmake_tsv(bif_tsv, src_outfp, tgt_outfp)

    #cleanup
    os.remove(tsv)
    os.remove(bif_tsv)

    return {'src': src_outfp, 'tgt': tgt_outfp}

def biclean(
        src: str, tgt: str, output_dir: str, 
        src_lang: str, tgt_lang: str, 
        **kwargs
    ) -> dict:
    src_outfp = os.path.join(output_dir, os.path.basename(src)) + '.bicleaned'
    tgt_outfp = os.path.join(output_dir, os.path.basename(tgt)) + '.bicleaned'

    #avoid overwriting/redoing work
    if os.path.exists(src_outfp) and os.path.exists(tgt_outfp):
        logger.info(f"Skipping; files already exist: {src_outfp} {tgt_outfp}")
        return {'src': src_outfp, 'tgt': tgt_outfp}

    #setup
    os.makedirs(output_dir, exist_ok=True)
    tsv = os.path.join(output_dir, os.path.basename(src)) + '.tsv'
    make_tsv(src, tgt, tsv)
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, f"{os.path.basename(tsv)}.bicleaned")
    annotated = os.path.join(output_dir, f'{os.path.basename(tsv)}.annotated')

    #NOTE: bicleaner already uses multiprocessing on a single tsv file 
    cmd = f"python {ROOT_DIR}/bifixclean/bicleaner_hardrules.py"\
          f" {tsv} {annotated} -s {src_lang} -t {tgt_lang}"\
          f" --scol 1 --tcol 2 --annotated_output"
    logger.info('RUNNING: ' + cmd)
    subprocess.call(cmd, shell=True)
    
    #extract the desired pairs
    cmd = f'cat {annotated}  | grep "keep$" | cut -f1,2 > {output}'
    logger.info('RUNNING: ' + cmd)
    subprocess.call(cmd, shell=True)

    unmake_tsv(output, src_outfp, tgt_outfp)

    #cleanup
    os.remove(tsv)
    os.remove(annotated)
    
    return {'src': src_outfp, 'tgt': tgt_outfp}

def langcheck(
        src: str, tgt: str, output_dir: str, 
        src_lang: str, tgt_lang: str, 
        aliases=None, 
        **kwargs
    ) -> dict:
    src_outfp = os.path.join(output_dir, os.path.basename(src)) + '.langfilter'
    tgt_outfp = os.path.join(output_dir, os.path.basename(tgt)) + '.langfilter'
    aux_outfp = os.path.join(output_dir, os.path.basename(src)) + '.langids'

    #avoid overwriting/redoing work
    if os.path.exists(src_outfp) and os.path.exists(tgt_outfp) and os.path.exists(aux_outfp):
        logger.info(f"Skipping; files already exist: {src_outfp} {tgt_outfp} {aux_outfp}")
        return {'src': src_outfp, 'tgt': tgt_outfp, 'langids': aux_outfp}

    #allow langcheck to also look for other aliases for this lang
    accepted_langs = [[src_lang], [tgt_lang]]
    if aliases:
        for i, lang in enumerate([src_lang, tgt_lang]):
            if lang in aliases:
                accepted_langs[i].extend(aliases[lang])

    #parallel process fasttext_langid, because it's not parallelized
    part = partial(
        fasttext_langid.main, 
        outdir=output_dir, 
        accepted_langs=accepted_langs
    )
    lng_src_fp, lng_tgt_fp, pred_fp = pll_multi(
        [src, tgt], 
        part, 
        n_jobs=CPU_COUNT, 
        outdir=output_dir, 
        output_name=os.path.basename(src) + '.langfilter'
    ) 
    shutil.move(lng_src_fp, src_outfp)
    shutil.move(lng_tgt_fp, tgt_outfp)
    shutil.move(pred_fp, aux_outfp)

    return {'src': src_outfp, 'tgt': tgt_outfp, 'langids': aux_outfp}

def _tagprotect(
        input_fps: list, #make it a list of 1 just to use it in pll_multi
        output_dir: str, 
        templates: list, 
        regex, 
        max_id=11, 
        dropout=0.1
    ) -> dict:
    r"""Replace URLs, emails, xml tags, etc. with tokens from templates."""
    input_fp = input_fps[0] #a list of length 1 (for use w/ pll_multi)

    output_fp = os.path.join(
        output_dir, os.path.basename(input_fp)) + '.tagprotect'
    repls_fp = output_fp + '.tag_repls'
    with open(input_fp, 'r', encoding='utf-8') as infile, \
            open(output_fp, 'w', encoding='utf-8') as outfile, \
            open(repls_fp, 'w', encoding='utf-8') as repls_fh:
        for line in infile:
            repls = []
            line = line.strip()
            matches = re.findall(regex, line)
            for i, match in enumerate(matches):
                if len(matches) == 1:
                    c = random.randrange(0, max_id)
                elif i > max_id:
                    c = '' #if maxed use a bare repl e.g. [TAG] by itself
                else:
                    c = i
                for j, extraction in enumerate(match):
                    if extraction and random.uniform(0,1) > dropout:
                        repl = templates[j].format(c)
                        line = line.replace(extraction, repl, 1)
                        repls.append([repl, extraction])
            repls_json = json.dumps(repls, ensure_ascii=False)

            repls_fh.write(repls_json + os.linesep)
            outfile.write(line + os.linesep)
    return [output_fp, repls_fp]

def tagprotect(
        src: str, tgt: str, output_dir: str, 
        src_lang: str, tgt_lang: str, 
        max_id=11, dropout=0.1, **kwargs
    ) -> dict:
    r"""Replace URLs, emails, xml tags, etc. with tokens from templates."""
    src_outfp = os.path.join(output_dir, os.path.basename(src)) + '.tagprotect'
    tgt_outfp = os.path.join(output_dir, os.path.basename(tgt)) + '.tagprotect'
    src_aux_outfp = os.path.join(output_dir, os.path.basename(src)) + '.tagprotect_repls'
    tgt_aux_outfp = os.path.join(output_dir, os.path.basename(tgt)) + '.tagprotect_repls'

    #avoid overwriting/redoing work
    if (
        os.path.exists(src_outfp) and 
        os.path.exists(tgt_outfp) and 
        os.path.exists(src_aux_outfp) and
        os.path.exists(tgt_aux_outfp) 
    ):
        logger.info(f"Skipping; files already exist: {src_outfp} {tgt_outfp}")
        return {
            'src': src_outfp, 'tgt': tgt_outfp, 
            'src_repls': src_aux_outfp, 'tgt_repls': tgt_aux_outfp
        }

    #parallel process the two files using _tagprotect
    part = partial(
        _tagprotect,
        output_dir=output_dir,
        templates=retagger.TEMPL, 
        regex=retagger.REGEX,
        max_id=max_id, 
        dropout=dropout,
    )
    src_fp, src_repls_fp = pll_multi(
        [src], 
        part, 
        n_jobs=CPU_COUNT, 
        outdir=output_dir, 
        output_name=os.path.basename(src) + '.tagprotect'
    )
    part = partial(
        _tagprotect,
        output_dir=output_dir,
        templates=retagger.TEMPL, 
        regex=retagger.REGEX,
        max_id=max_id, 
        dropout=dropout,
    )
    tgt_fp, tgt_repls_fp = pll_multi(
        [tgt], 
        part, 
        n_jobs=CPU_COUNT, 
        outdir=output_dir, 
        output_name=os.path.basename(tgt) + '.tagprotect'
    )

    #rename files so they have predictable/consistent naming
    shutil.move(src_fp, src_outfp)
    shutil.move(tgt_fp, tgt_outfp)
    shutil.move(src_repls_fp, src_aux_outfp)
    shutil.move(tgt_repls_fp, tgt_aux_outfp)

    return {
        'src': src_outfp, 'tgt': tgt_outfp, 
        'src_repls': src_aux_outfp, 'tgt_repls': tgt_aux_outfp
    }

def dedup(
        src: str, tgt: str, output_dir: str, 
        src_lang: str, tgt_lang: str, 
        **kwargs
    ) -> dict:
    src_outfp = os.path.join(output_dir, os.path.basename(src)) + '.dedup'
    tgt_outfp = os.path.join(output_dir, os.path.basename(tgt)) + '.dedup'

    #avoid overwriting/redoing work
    if os.path.exists(src_outfp) and os.path.exists(tgt_outfp):
        logger.info(f"Skipping; files already exist: {src_outfp} {tgt_outfp}")
        return {'src': src_outfp, 'tgt': tgt_outfp}

    os.makedirs(output_dir, exist_ok=True)
    tsv = os.path.join(output_dir, os.path.basename(src)) + '.tsv'
    make_tsv(src, tgt, tsv)

    out_tsv = os.path.join(output_dir, os.path.basename(src)) + '.tsv.dedup'
    cmd = f"sort -u {tsv} -o {out_tsv}"
    logger.info('RUNNING: ' + cmd)
    subprocess.call(cmd, shell=True)

    unmake_tsv(out_tsv, src_outfp, tgt_outfp)

    #cleanup
    os.remove(tsv)
    os.remove(out_tsv)

    return {'src': src_outfp, 'tgt': tgt_outfp}
