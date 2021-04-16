import joblib
import logging
import math
import os
import subprocess
import shutil
import tempfile
import time
from joblib import Parallel, delayed
from typing import *

from tqdm import tqdm

logger = logging.getLogger('parallely')
logger.setLevel(logging.INFO)

class PllError(BaseException):
    """Raise for errors in multiprocessing parallel files parallely."""

def get_file_length(filepath: str):
    r"""
    Run linux awk to count number of records (faster than doing it in python).

    NOTE: Use awk instead of wc because wc only counts \n, while awk counts
    records like other tools do, see: https://stackoverflow.com/a/35052861
    """
    out = subprocess.check_output(["awk", "END{print NR}", filepath])
    length = int(out.split()[0])
    return length

def pll_multi(
        parallel_files: list, 
        function: Callable, 
        n_jobs: Optional[int]=1,
        outdir: Optional[str]=None,
        output_name: Optional[str]='output',
        remove: Optional[bool]=True,
    ):
    r"""
    Process parallel files parallely using joblib to split `parallel_files` 
    into subfiles, invoke `function` to create outputs for each subfile,
    and then concatenate the outputs into the final output.

    Args:
        parallel_files: a list of parallel files, e.g. for machine translation
        function: the callable to invoke parallely
        n_jobs: the number of jobs (CPU processes) to use, default=1
        outdir: directory to put the result, default=first file's dirname 
        output_name: the basename to call the output files
        remove: remove files that are generated along the way, default=True

    Returns:
        catted: output files resulting from the function

    NOTE: The function must take a list of files as input and return a list of
          files as output (lists of length 0 or 1 are allowed).
    """
    if outdir is None:
        outdir = os.path.dirname(parallel_files[0])
    os.makedirs(outdir, exist_ok=True)

    temp_dir_obj = tempfile.TemporaryDirectory(dir=outdir)
    temp_dir = temp_dir_obj.name 
    
    length = get_file_length(parallel_files[0])
    if not length:
        raise PllError(f"cannot multiprocess empty file(s): {parallel_files}")
    lines_per_job = math.floor(length / n_jobs) 
    if not lines_per_job:
        lines_per_job = length

    lists_of_splits = []
    for i, fp in enumerate(parallel_files):
        ident = f"file{i}"
        subname = os.path.join(temp_dir, ident + ".tmp.") 
        command = f"split -l {lines_per_job} {fp} {subname}"
        subprocess.call(command, shell=True)
        subfiles = [os.path.join(temp_dir, piece) 
                    for piece in sorted(os.listdir(temp_dir))
                    if ident in piece]
        lists_of_splits.append(subfiles)

    zipped = zip(*lists_of_splits) #[(inp0_0, inp1_0, ...), ...]
    result = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(function)(item) for item in zipped
    )                              #[[out0_0, out1_0, ...], ...]
    outs = list(zip(*result))      #[(out0_0, out0_1, ...), ...]

    catted = []
    for i, out in enumerate(outs):
        suffix = '.' + str(i)
        catted_fp = os.path.join(outdir, output_name + suffix)
        command = f'cat {" ".join(out)} > {catted_fp}'
        subprocess.call(command, shell=True)
        if remove:
            for o in out:
                os.remove(o)
        catted.append(catted_fp)

    temp_dir_obj.cleanup()
    return catted

def pll_single(
        fp: str, 
        function: Callable, 
        n_jobs: Optional[int]=1,
        outdir: Optional[str]=None,
        output_name: Optional[str]='output',
        remove: Optional[bool]=True,
    ):
    r"""
    Process a file parallely using joblib to split the file `fp` 
    into subfiles, invoke `function` to create outputs for each subfile,
    and then concatenate the outputs into the final output.

    Args:
        fp: a file to work on 
        function: the callable to invoke parallely
        n_jobs: the number of jobs (CPU processes) to use, default=1
        outdir: directory to put the result, default=file's dirname
        output_name: the basename to call the output files
        remove: remove files that are generated along the way, default=True

    Returns:
        catted: output file resulting from the function
    """
    if outdir is None:
        outdir = os.path.dirname(fp)
    os.makedirs(outdir, exist_ok=True)

    temp_dir_obj = tempfile.TemporaryDirectory(dir=outdir)
    temp_dir = temp_dir_obj.name 
    
    length = get_file_length(fp)
    if not length:
        raise PllError(f"cannot multiprocess empty file: {fp}")
    lines_per_job = math.floor(length / n_jobs) 
    if not lines_per_job:
        lines_per_job = length

    subname = os.path.join(temp_dir, "piece.tmp.") 
    command = f"split -l {lines_per_job} {fp} {subname}"
    subprocess.call(command, shell=True)
    subfiles = [os.path.join(temp_dir, piece) 
                for piece in sorted(os.listdir(temp_dir))
                if 'piece' in piece]

    result = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(function)(item) for item in subfiles 
    )

    catted_fp = os.path.join(outdir, output_name + '.out')
    command = f'cat {" ".join(result)} > {catted_fp}'
    subprocess.call(command, shell=True)
    if remove:
        for o in result:
            os.remove(o)

    temp_dir_obj.cleanup()
    return catted_fp
