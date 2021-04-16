import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import *
from functools import partial

import fasttext

FASTTEXT_MODEL_URL = "http://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

logger = logging.getLogger('fasttext_langid')
logger.setLevel(logging.INFO)

def get_file_length(filepath: str):
    r"""
    Run linux awk to count number of records (faster than doing it in python).

    NOTE: Use awk instead of wc because wc only counts \n, while awk counts
    records like other tools do, see: https://stackoverflow.com/a/35052861
    """
    out = subprocess.check_output(["awk", "END{print NR}", filepath])
    length = int(out.split()[0])
    return length

def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, 
            e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded 
            file should start with `hash_prefix`.Default: None
        progress (bool, optional): whether or not to display a progress bar 
            Default: True

    Acknowledgements:
        https://github.com/pytorch/pytorch/blob/master/torch/hub.py
    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def download_model(outdir=None):
    if outdir is None:
        outdir = os.getcwd()
    model = os.path.join(outdir, FASTTEXT_MODEL_URL.split("/")[-1])
    if not os.path.isfile(model):
        os.makedirs(outdir, exist_ok=True)
        download_url_to_file(FASTTEXT_MODEL_URL, model)
    return model

def accepted_line(line: str, accepted_langs: list, model: object, k: int):
    r"""
    Return True if the line is predicted to be from one of the accepted_langs. 
    """
    langs, scores = model.predict(line, k)
    preds = set([l.replace("__label__", "") for l in langs])
    accepted = False
    for accepted_lang in accepted_langs:
        if accepted_lang in preds:
            accepted = True
            break
    return accepted, list(preds)

def check_lang(
        fps: list, 
        outdir: str,
        fasttext_model: str, 
        accepted_langs: List[list], 
        k: Optional[int]=3, 
    ):
    r"""
    Create new files in the outdir which have only the lines from the fps
    that match the accepted_langs. If the fps are parallel files, then ALL
    translations from all fps must match to be written out.
    """
    if outdir is None:
        outdir = os.path.dirname(parallel_files[0])
    model = fasttext.load_model(fasttext_model)
    length = get_file_length(fps[0])
    out_fps = [os.path.join(outdir, os.path.basename(fp)) + '.langcheck' 
                for fp in fps]
    inp_fhs = [open(fp, 'r', encoding='utf-8') for fp in fps]
    out_fhs = [open(fp, 'w', encoding='utf-8') for fp in out_fps]
    pred_fp = os.path.join(outdir, os.path.basename(fps[0])) + '.langid'

    with open(pred_fp, 'w', encoding='utf-8') as pred_fh:
        for i in range(length):
            lines = [fh.readline().strip() for fh in inp_fhs]
            write = True
            preds = []
            for j, line in enumerate(lines):
                accepted, pred = accepted_line(
                    line, accepted_langs[j], model, k)
                preds.append(pred)
                if not accepted:
                    write = False
            if write:
                for j, line in enumerate(lines):
                    out_fhs[j].write(line + os.linesep)
            pred_fh.write(json.dumps(preds, ensure_ascii=False) + '\n')

    for i in range(len(inp_fhs)):
        inp_fhs[i].close()
        out_fhs[i].close()
    pred_fh.close()

    out_fps.append(pred_fp)
    return out_fps

def main(parallel_files: list, outdir: str, accepted_langs: List[list]):
    model = download_model()
    output = check_lang(parallel_files, outdir, model, accepted_langs)
    logger.info(f"Created lang checked files: {output}")
    return output

if __name__ == "__main__":
    import fire
    fire.Fire(main)

"""
Cite this when using the language identification model from fastText above: 

[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification

@article{joulin2016bag,
    title={Bag of Tricks for Efficient Text Classification},
    author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
    journal={arXiv preprint arXiv:1607.01759},
    year={2016}
}

[2] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, FastText.zip: Compressing text classification models

@article{joulin2016fasttext,
    title={FastText.zip: Compressing text classification models},
    author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
    journal={arXiv preprint arXiv:1612.03651},
    year={2016}
}
"""