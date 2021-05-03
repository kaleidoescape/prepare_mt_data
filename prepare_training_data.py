import argparse
import collections.abc
import json
import logging
import os
import shutil
import subprocess
import textwrap
from typing import *

import yaml
from pydantic import BaseModel, validator

import clean
import create_queries
import utils

COMMANDS = {
    'charfix': clean.charfix,
    'bifix': clean.bifix,
    'biclean': clean.biclean,
    'dedup': clean.dedup,
    'langcheck': clean.langcheck,
    'tagprotect': clean.tagprotect,
    'queries': create_queries.create_queries,
}

#be explicit, so that logging occurs even if this is run as main
logger = logging.getLogger('prepare_training_data')
logger.setLevel(logging.INFO)


def main(
        pconfig: str,
        outdir: str,
        command: str,
        name: Optional[str]='',
    ):
    r"""
    Preprocess training data with a text processor.

    Args:
        config: config file with test sets
        outdir: output directory to put intermediate files into
        text_processor: string name of the TextProcessor class to use
        name: optional string to name results file

    Example config:
        { data: { 
                my_data_set1: {
                    src_lang: xx,
                    tgt_lang: yy,
                    src: source_file
                    tgt: target_file 
                },
                my_data_set2: {...}
            }
        }
    """
    os.makedirs(outdir, exist_ok=True)

    func = COMMANDS[command]

    results = {'data': {}}
    for k in pconfig.data:
        this_outdir = os.path.join(outdir, k)

        os.makedirs(this_outdir, exist_ok=True)
        src_lang, tgt_lang = pconfig.data[k].src_lang, pconfig.data[k].tgt_lang
        src_data, tgt_data = pconfig.data[k].src, pconfig.data[k].tgt
        src_data_out = os.path.join(this_outdir, os.path.basename(src_data))
        logger.info(f"Processing {name} {k} in {this_outdir}...")

        func_args = pconfig.data[k].dict()
        func_args.update(**pconfig.args)
        func_args['output_dir'] = this_outdir
        outputs = func(**func_args)

        results['data'][k] = {
            'src_lang': src_lang, 
            'tgt_lang': tgt_lang
        }
        results['data'][k].update(outputs)

    if name:
        results_fp = os.path.join(outdir, f'config.{command}.{name}.yml')
    else:
        results_fp = os.path.join(outdir, f'config.{command}.results.yml')
    with open(results_fp, 'w', encoding='utf-8') as results_fh:
        yaml.dump(results, results_fh)

    return results_fp

def parse_args():
    epilog = textwrap.dedent(f"""
    Example config.yml:
    -------------------
    data:
      my_test_set:
        src_lang: xx
        tgt_lang: yy
        src: src_file_path
        tgt: tgt_file_path
    """)
    
    parser = argparse.ArgumentParser(
        formatter_class=Formatter,
        description="Generate data using the requested text processors for source and target.",
        epilog=epilog
    )
    parser.add_argument('--configs', nargs='+', required=True,
        help='path to yaml config file(s) with source/target data (later override earlier)')
    parser.add_argument('--outdir', required=True,
        help="output directory where to save the evaluation results directory")
    parser.add_argument('--command', choices=list(COMMANDS.keys()), required=True,
        help='the name of the text processor to use')
    parser.add_argument('--name', default='results', 
        help="optional name for results file prefix")
    args, rest = parser.parse_known_args()
    args.rest = rest 
    
    logger.info(f"Using {args.command} with {args.configs} in {args.outdir}...")

    args.config = utils.parse_configs(configs, args.outdir)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(
        args.config,
        outdir=args.outdir,
        command=args.command,
        name=args.name,
    )
