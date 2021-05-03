import argparse
import collections.abc
import json
import logging
import os
import shutil
import subprocess
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
    os.makedirs(outdir, exist_ok=True)
    pconfig = {}
    for config in configs:
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
