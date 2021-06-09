#!/usr/bin/env python3.7
r"""
Translate documents stored inside of an xml file with <doc> tags, line-by-line.
Use a supplied command to invoke a translator subprocess. Any command can be
used (including piping together more complicated commands), as long as it is 
passed in to the program as a string.

Example for SentencePiece with Fairseq (note the double quotes around the 
entire command, and the therefore the single quotes used for grep):

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

import sentencepiece as spm
import xml.etree.ElementTree as ET
logger = logging.getLogger(__name__)

def _mux(docs: list, process_stdin: IO, q: queue.Queue):
    for i, doc in enumerate(docs):
        count = 0
        sents = doc.text.strip().split('\n')
        for line in sents:
            line = line + '\n'
            process_stdin.write(line.encode('utf-8'))
            count += 1
        q.put((i, count))
    q.put(None) #poison
    process_stdin.close()

def _demux(docs: list, process_stdout: IO, q: queue.Queue):
    while True:
        item = q.get()
        if item is None:
            break
        i, count = item
        new_doc = []
        for l in range(count):
            line = process_stdout.readline().decode('utf-8')
            new_doc.append(line)
        docs[i].text = '\n' + ''.join(new_doc) #add \n after opening <doc> tag
        q.task_done()
    return True

def mux_demux(docs: list, process_stdin: IO, process_stdout: IO):
    q = queue.Queue()
    muxer = threading.Thread(target=_mux, args=(docs, process_stdin, q))
    demuxer = threading.Thread(target=_demux, args=(docs, process_stdout, q))
    muxer.start()
    demuxer.start()
    muxer.join()
    demuxer.join()

def main(infp: str, outfp: str, subcommand: list):
    tree = ET.parse(infp) #WARNING! Reading the whole file into memory
    root = tree.getroot()
    docs = root.findall('doc')

    process = subprocess.Popen(
        subcommand, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=sys.stderr,
        shell=True,
    )
    mux_demux(docs, process.stdin, process.stdout) #edit docs in-place
    process.kill() #need to kill fairseq-interactive, else it waits forever

    with open(outfp, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)


def parse_args():
    r"""Parse command line args for process_folder.py"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        description="Translate xml file with <doc> using a shell subprocess.",
        epilog="Example usage: translate_xml_docs.py input.xml output.xml ~/marian-dev/build/marian-decoder --devices 0 1"
    )
    parser.add_argument('input', 
        help="the input file with <doc> tags that have text in them")
    parser.add_argument('output', 
        help="the output file to create")
    parser.add_argument('command', 
        help="the translation command to run as a string")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args.input, args.output, args.command) 


