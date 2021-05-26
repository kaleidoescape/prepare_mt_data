import argparse
import collections.abc
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
from typing import *
from typing import IO #above * doesn't get this one

import yaml
from pydantic import BaseModel, validator

class ArgParseHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter, 
        argparse.RawDescriptionHelpFormatter
    ): 
    r"""Format help text in the arg parser in a prettier way."""
    pass

class DatasetArg(BaseModel):
    r"""A data set field from the config file."""
    src_lang: str
    tgt_lang: str
    src: str 
    tgt: str 

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

def update_dict_recursively(d, u):
    r"""
    Normal dict.update() results in nested dicts being overwritten wholesale.
    We need to update the keys inside of the nested dicts as well. 
    """
    #https://stackoverflow.com/a/3233356
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict_recursively(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def parse_configs(configs, outdir=None):
    pconfig = {}
    for config in configs:
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            config_bck = os.path.join(outdir, os.path.basename(config)) 
            os.makedirs(os.path.dirname(config_bck), exist_ok=True)
            if not os.path.exists(config_bck):
                shutil.copyfile(config, config_bck)
        with open(config, 'r', encoding='utf-8') as infile:
            c = yaml.safe_load(infile)
            pconfig = update_dict_recursively(pconfig, c)
    if outdir:
        with open(os.path.join(outdir, 'config.last_input.yml'), 'w', encoding='utf-8') as outfile:
            yaml.dump(pconfig, outfile)

    config = ConfigArgs.parse_obj(pconfig).dict()
    return config

def get_file_length(filepath):
    r"""
    Run linux awk to count number of records (faster than doing it in python).

    NOTE: Use awk instead of wc because wc only counts \n, while awk counts
    records like other tools do, see: https://stackoverflow.com/a/35052861
    """
    out = subprocess.check_output(["awk", "END{print NR}", filepath])
    length = int(out.split()[0])
    return length

def write_json_line(data: Union[dict, list], fh: IO):
    r"""Write json-compatible data into the fh stream."""
    s = json.dumps(data, ensure_ascii=False, sort_keys=True)
    fh.write(s + '\n')

def get_line_offsets(fh):
    """
    Create a dict of line numbers to their byte offset. After this, you can 
    read a arbitrary line i with: file.seek(line_offsets[i]); file.readline()
    """
    line_offsets = {0: 0} 
    for i, line in enumerate(fh):
        if line != '': #this is EOF
            line_offsets[i+1] = fh.tell()
    fh.seek(0)
    return line_offsets

def setup_logger(
        name='scripts', 
        folder=None, 
        level=logging.DEBUG, 
        to_stdout=True
    ):
    r"""Write logs to the file at folder/name.log. Default folder is cwd."""
    if folder is None:
        folder = os.getcwd()
    os.makedirs(folder, exist_ok=True)
    fp = os.path.join(folder, name + '.log')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    #delay is a half-fix for mem leak: https://bugs.python.org/issue23010
    file_handler = logging.FileHandler(fp, delay=True) 
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('[%(asctime)s:%(levelname)s:%(name)s:%(lineno)d] %(message)s')
    file_handler.setFormatter(file_formatter)

    stream_handler = None
    if to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('[%(levelname)s:%(name)s:%(lineno)d] %(message)s')
        stream_handler.setFormatter(stream_formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        if to_stdout:
            logger.addHandler(stream_handler)

    return logger
