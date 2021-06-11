#!/usr/bin/env python3.7
r"""
Translate documents stored inside of an xml file with <doc> tags, line-by-line.
Use a supplied command to invoke a translator subprocess. Any command can be
used (including piping together more complicated commands), as long as it is 
passed in to the program as a string.

Example for SPM -> Fairseq -> grep -> SPM (note the double quotes around the
entire command making it a string, and the single quotes used for grep):

CUDA_VISIBLE_DEVICES=0 python translate_xml_docs.py \
    input.xml \
    output.xml \
    "~/sentencepiece/build/src/spm_encode --model=bpe.model --output_format=piece \
    | fairseq-interactive data-bin/ --path checkpoint_best.pt --<fairseq params> \
    | grep -P 'D-[0-9]+' | cut -f3 \
    | ~/sentencepiece/build/src/spm_decode --model bpe.model"

"""
import argparse
import logging
import os
import queue
import subprocess
import sys
import textwrap
import threading
from typing import *
from typing import IO #the * above won't load this

import xml.etree.ElementTree as ET
logger = logging.getLogger(__name__)

def _mux(docs: list, process_stdin: IO, q: queue.Queue):
    r"""
    Read a list of docs, where each doc is a list of text snippets. Separate
    the text snippets into sentences by breaking them on newlines. Send
    sentences to the subprocess's stdin, one-by-one.
    """
    for i, doc in enumerate(docs):
        count = 0
        sents = doc.strip().split('\n')
        for line in sents:
            line = line + '\n'
            process_stdin.write(line.encode('utf-8'))
            count += 1
        q.put((i, count))
    q.put(None) #poison
    process_stdin.close()

def _demux(docs: list, process_stdout: IO, q: queue.Queue):
    r"""
    Read data from the subprocess's stdout and update the docs list in-place.
    """
    while True:
        item = q.get()
        if item is None:
            break
        i, count = item
        sents = []
        for l in range(count):
            line = process_stdout.readline().decode('utf-8').strip()
            sents.append(line)
        docs[i] = '\n'.join(sents)
        q.task_done()
    return True

def mux_demux(docs: list, process_stdin: IO, process_stdout: IO):
    r"""
    Start the mux and demux threads which send data to the subproccess's stdin
    and read the correct amount of data from the subprocess's stdout.
    """
    q = queue.Queue()
    muxer = threading.Thread(target=_mux, args=(docs, process_stdin, q))
    demuxer = threading.Thread(target=_demux, args=(docs, process_stdout, q))
    muxer.start()
    demuxer.start()
    muxer.join()
    demuxer.join()

def extract_texts(docs):
    r"""
    Convert text snippets from doc and p tags into a list. The list can 
    be shared with the threads. 
    """
    subdocs = []
    for doc in docs:
        subdocs.append(doc.text)
    return subdocs

def replace_texts(docs, subdocs):
    r"""
    Replace the translated text snippets into the xml docs, taking care to
    replace the text inside internal paragraph tags as well.
    """
    for i, doc in enumerate(docs):
        doc.text = '\n' + subdocs[i] #newline after opening tag

def main(infp: str, outfp: str, subcommand: str):
    r"""
    Translate the infp xml file into the outfp xml file by feeding the text
    inside the xml <doc>s (and their <p>s) into the translation subcommand.
    """
    process = subprocess.Popen( #start translator subprocess in subshell
        subcommand, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=sys.stderr,
        shell=True,
    )

    tree = ET.parse(infp) #WARNING! Reading the whole file into memory
    root = tree.getroot()
    docs = root.findall('doc')

    subdocs = extract_texts(docs)
    mux_demux(subdocs, process.stdin, process.stdout)
    replace_texts(docs, subdocs)

    process.kill() #kill translator, otherwise it waits forever 

    with open(outfp, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)


def parse_args():
    r"""Parse command line args for process_folder.py"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        description="Translate xml file with <doc> using a shell subprocess.",
        epilog="Example usage: translate_xml_docs.py input.xml output.xml ~/marian-dev/build/marian-decoder --devices 0 1"
    )
    parser.add_argument('input', type=str,
        help="the input file with <doc> tags that have text in them")
    parser.add_argument('output', type=str, 
        help="the output file to create")
    parser.add_argument('command', type=str,
        help="the translation command to run as a string")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args.input, args.output, args.command) 


