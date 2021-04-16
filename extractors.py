"""
Extract parallel source/target language files for machine translation
from other common parallel formats.
"""
import argparse
import os
import re
from typing import *

import xml.etree.ElementTree as ET

class ExtractorError(BaseException):
    """Raise for errors extracting parallel data from files."""

def fix_broken_chars(text: str) -> str:
    text = text.replace('\x00', '')
    text = text.replace('\u200c', '')
    text = text.replace('\r', '')
    return text

def extract_tmx(inp_fp: str, output_name: str, src_lang: str, tgt_lang: str):
    r"""
    Extract parallel files from a tmx file. 

    Args:
        inp_fp: input .tmx file
        output_name: output filename (including output directory)
        src_lang: source language name (will be appended to output_name) 
        tgt_lang: target language name (will be appended to output_name)
    """
    #About namespace expansion: https://stackoverflow.com/a/44283775
    ns = r'{http://www.w3.org/XML/1998/namespace}' 
    src_fp = f"{output_name}.{src_lang}"
    tgt_fp = f"{output_name}.{tgt_lang}"
    os.makedirs(os.path.dirname(src_fp), exist_ok=True)
    skipped = 0
    total = 0
    tree = ET.parse(inp_fp)
    root = tree.getroot()
    body = root.find('body')
    with open(inp_fp, 'r', encoding='utf-8') as infile, \
         open(src_fp, 'w', encoding='utf-8') as src_fh, \
         open(tgt_fp, 'w', encoding='utf-8') as tgt_fh:
        for i, tu in enumerate(body.findall('tu')):
            total += 1
            src_seg, tgt_seg = None, None
            tuvs = tu.findall('tuv')
            for tuv in tuvs:
                lang_key = '{}lang'.format(ns)
                if tuv.attrib[lang_key] == src_lang:
                    src_seg = tuv.find('seg')
                elif tuv.attrib[lang_key] == tgt_lang:
                    tgt_seg = tuv.find('seg')
            if src_seg is not None and tgt_seg is not None:
                src_text = fix_broken_chars(src_seg.text)
                tgt_text = fix_broken_chars(tgt_seg.text)
                #must have the same number of lines on src/tgt sides
                src_text = src_text.replace('\n', ' ')
                tgt_text = tgt_text.replace('\n', ' ')
                if src_text and tgt_text:
                    src_fh.write(src_text + os.linesep)
                    tgt_fh.write(tgt_text + os.linesep)
                else:
                    skipped += 1
            else:
                skipped += 1
    print(f"Skipped {skipped}/{total} degenerate lines in {inp_fp}")


def extract_tsv(
        inp_fp: str, 
        output_name: str, 
        src_lang: str, 
        tgt_lang: str, 
        src_col: Optional[int]=0, 
        tgt_col: Optional[int]=1
    ):
    r"""
    Extract parallel files from a tsv file.

    Args:
        inp_fp: input .tsv file
        output_name: output filename (including output directory)
        src_lang: source language name (will be appended to output_name) 
        tgt_lang: target language name (will be appended to output_name)
        src_col: the tsv column of the source language (default=0)
        tgt_col: the tsv column of the target language (default=1)
    """
    src_fp = f"{output_name}.{src_lang}"
    tgt_fp = f"{output_name}.{tgt_lang}"
    skipped = 0
    total = 0
    with open(inp_fp, 'r', encoding='utf-8') as infile, \
         open(src_fp, 'w', encoding='utf-8') as src_fh, \
         open(tgt_fp, 'w', encoding='utf-8') as tgt_fh:
        for i, line in enumerate(infile):
            line = fix_broken_chars(line)
            split_line = line.strip().split('\t')
            try:
                src_line = split_line[src_col]
                tgt_line = split_line[tgt_col]
            except IndexError as e:
                print(f"Skipping line {i} due to error: {e}; {line}")
                skipped += 1
                continue
            src_fh.write(src_line + os.linesep)
            tgt_fh.write(tgt_line + os.linesep)
            total += 1
    print(f"Skipped {skipped}/{total} degenerate lines in {inp_fp}")

def extract_sgm(inp_fp: str, output_name: str, lang: str):
    r"""
    Extract all segments, i.e. <seg id="x"> entries, from a .sgm file. 

    Args:
        inp_fp: input .sgm file
        output_name: output filename (including output directory)
        lang: language name (will be appended to output_name) 
    """
    out_fp = f"{output_name}.{lang}"
    with open(inp_fp, 'r', encoding='utf-8') as sgm_fh, \
         open(out_fp, 'w', encoding='utf-8') as out_fh:
        for i, line in enumerate(sgm_fh):
            line = fix_broken_chars(line)
            line = line.strip()
            if line.startswith("<seg id=") and line.endswith("</seg>"):
                line = re.sub('<seg id="\d+">', '', line)
                line = re.sub('</seg>', '', line)
                out_fh.write(line + os.linesep)

def extract_sgms(
        src_sgm: str, 
        tgt_sgm: str, 
        output_name: str, 
        src_lang: str, 
        tgt_lang: str, 
    ):
    r"""
    Extract parallel plaintext files from parallel .sgm files.

    Args:
        src_sgm: source .sgm file
        tgt_sgm: source .sgm file
        output_name: output filename (including output directory)
        src_lang: source language name (will be appended to output_name) 
        tgt_lang: target language name (will be appended to output_name)
    """
    extract_sgm(src_sgm, output_name, src_lang)
    extract_sgm(tgt_sgm, output_name, tgt_lang)

def dedup(
        train_src: str, 
        train_tgt: str, 
        outdir: str, 
        test_srcs: list, 
        test_tgts: list
    ):
    """
    Remove test sentence pairs from the training data.

    Args:
        train_src: source sentence training data
        train_tgt: parallel target sentence training data
        outdir: output directory to save new files
        test_srcs: list of test and/or dev files for source sentences
        test_tgts: list of test and/or dev files for parallel target sentences
    """
    src_out_fp = os.path.join(outdir, os.path.basename(train_src) + '.dedup')
    tgt_out_fp = os.path.join(outdir, os.path.basename(train_tgt) + '.dedup')

    test_lines = set()
    src_fhs = [open(fp, 'r', encoding='utf-8') for fp in test_srcs] 
    tgt_fhs = [open(fp, 'r', encoding='utf-8') for fp in test_tgts]
    for j, src_fh in enumerate(src_fhs):
        tgt_fh = tgt_fhs[j]
        src_line = src_fh.readline().strip()
        tgt_line = tgt_fh.readline().strip()
        test_lines.add(f"{src_line}\t{tgt_line}")
    [fh.close() for fh in src_fhs]
    [fh.close() for fh in tgt_fhs]

    skipped = 0
    total = 0
    with open(train_src, 'r', encoding='utf-8') as src_fh, \
         open(train_tgt, 'r', encoding='utf-8') as tgt_fh, \
         open(src_out_fp, 'w', encoding='utf-8') as src_out_fh, \
         open(tgt_out_fp, 'w', encoding='utf-8') as tgt_out_fh:
        for i, src_line in enumerate(src_fh):
            total += 1
            src_line = src_line.strip()
            tgt_line = tgt_fh.readline().strip()
            both_line = f"{src_line}\t{tgt_line}"
            if both_line in test_lines:
                skipped += 1
                continue
            src_out_fh.write(src_line + os.linesep)
            tgt_out_fh.write(tgt_line + os.linesep)
    print(f"Skipped {skipped}/{total} duplicate lines.")

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser = parser.add_subparsers(dest='command')

    dedup_parser = subparser.add_parser('dedup')
    dedup_parser.add_argument('train_src', 
        help='the file with source language sentences')
    dedup_parser.add_argument('train_tgt', 
        help='the parallel file with target language sentences')
    dedup_parser.add_argument('--outdir', required=True,
        help='output directory where to put deduped files')
    dedup_parser.add_argument('--test_srcs', nargs='+', required=True,
        help="list of test files in source language")
    dedup_parser.add_argument('--test_tgts', nargs='+', required=True,
        help="parallel list of test files in target language")

    extract_tsv_parser = subparser.add_parser('extract_tsv')
    extract_tsv_parser.add_argument('input_fp',
        help='input tab-separated file of source and target sentences')
    extract_tsv_parser.add_argument('output_name',
        help='prefix of the output (src_lang/tgt_lang will be added as suffixes')
    extract_tsv_parser.add_argument('src_lang',
        help='source language (added as suffix to output_name)')
    extract_tsv_parser.add_argument('tgt_lang',
        help='target language (added as suffix to output_name)')
    extract_tsv_parser.add_argument('--src_col', type=int, default=0,
        help='the column of the tsv that has the source sentences')
    extract_tsv_parser.add_argument('--tgt_col', type=int, default=1,
        help='the column of the tsv that has the target sentences')

    extract_sgms_parser = subparser.add_parser('extract_sgms')
    extract_sgms_parser.add_argument('src_sgm', 
        help='the file with source language sentences')
    extract_sgms_parser.add_argument('tgt_sgm', 
        help='the parallel file with target language sentences')
    extract_sgms_parser.add_argument('output_name',
        help='prefix of the output (src_lang/tgt_lang will be added as suffixes')
    extract_sgms_parser.add_argument('src_lang',
        help='source language (added as suffix to output_name)')
    extract_sgms_parser.add_argument('tgt_lang',
        help='target language (added as suffix to output_name)')

    extract_sgms_parser = subparser.add_parser('extract_tmx')
    extract_sgms_parser.add_argument('tmx', 
        help='the file with tmx and sentences in <seg></seg>')
    extract_sgms_parser.add_argument('output_name',
        help='prefix of the output (src_lang/tgt_lang will be added as suffixes')
    extract_sgms_parser.add_argument('src_lang',
        help='source language (added as suffix to output_name)')
    extract_sgms_parser.add_argument('tgt_lang',
        help='target language (added as suffix to output_name)')

    args = parser.parse_args()
    return args

def main(args):
    if args.command == 'dedup':
        dedup(args.train_src, args.train_tgt, args.outdir, args.test_srcs, args.test_tgts)
    if args.command == 'extract_tsv':
        extract_tsv(args.input_fp, args.output_name, args.src_lang, args.tgt_lang, args.src_col, args.tgt_col)
    if args.command == 'extract_sgms':
        extract_sgms(args.src_sgm, args.tgt_sgm, args.output_name, args.src_lang, args.tgt_lang)
    if args.command == 'extract_tmx':
        extract_tmx(args.tmx, args.output_name, args.src_lang, args.tgt_lang)

if __name__ == '__main__':
    args = parse_args()
    main(args)
