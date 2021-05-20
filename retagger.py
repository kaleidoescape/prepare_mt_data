#!/usr/bin/env python3.7
# -*- coding: utf-8 -*
"""
"""
import json
import logging
import re
import sys
from typing import *

logger = logging.getLogger('retagger')
logger.setLevel(logging.INFO)

#We pass URLs, emails, etc. through MT unscathed by wrapping them in a
#a special tag protector symbol (which is not BPEd), and then re-inserting
#them back into the translation on the output side. Find them using regex:
#
# TAG regex source (Svetlana Tchistiakova): 
# https://gist.github.com/kaleidoescape/524f6f53a4562eaf6d8f1463f4d54670
TAG_REGEX = r"(?:\s*<(?:[A-Za-z]+|/)[^<]*?>\s*)"
TAG_TEMPL = ' [TAG{}] '
#
# URL regex source (Mathias Bynens / Gruber) @gruber (71 chars) version:
# https://mathiasbynens.be/demo/url-regex
# Note: I was using https://gist.github.com/gruber/8891611, but it's too inefficient
URL_REGEX = r"#\b(?:(?:[\w-]+://?|www[.])[^\s()<>]+(?:\([\w\d]+\)|(?:[^[:punct:]\s]|/)))#iS"
URL_TEMPL = ' [URL{}] '
#
# EMAIL regex source (Regular-Expressions.info) 2nd to last one: 
# http://www.regular-expressions.info/email.html
EMAIL_REGEX = r"(?:[^\w][a-z0-9!#$%&'*+/=?^_‘{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_‘{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?[^\w])"
EMAIL_TEMPL = ' [EML{}] '
#
#We OR the regexes together so that we get an ordered list of extractions
#(order could be important metadata in the future, which we would lose if
#we applied the regexes one at a time)
#NOTE: To OR the regexes together the order matters, and above regexes
#should be surrounded by non-capture groups, and it is recommended to
#surround the pattern with spaces or non-word chars for best results.
full_regex = f"({TAG_REGEX})|({URL_REGEX})|({EMAIL_REGEX})"
REGEX = re.compile(full_regex)
TEMPL = (TAG_TEMPL, URL_TEMPL, EMAIL_TEMPL) #same order as ORed regexes

def extract_tags(text):
    r"""
    Extract urls/emails/tags/etc from the text and return a tuple
    of the cleaned up text and a list of [(symbol, url/email/etc)].
    """
    matches = re.findall(REGEX, text)
    tags = []
    for i, match in enumerate(matches):
        for j, m in enumerate(match):
            if m:
                repl = TEMPL[j].format(i)
                text = text.replace(m, repl, 1)
                tags.append([repl, m])
    return text, tags 

def reinsert_tags(text, tags):
    r"""
    Reinsert urls/emails/tags/etc. into the text using the list of
    [(symbol, url/email/etc)]. In case MT failed to add one of
    the symbols into the output, the url/email will just be
    appended to the end of the text.
    """
    for tag, item in tags:
        tag = tag.strip()
        if tag in text:
            text = text.replace(tag, item, 1)
        else:
            #MT didn't output it, but we don't want to lose it
            text = f"{text} {item}"
    return text

def extract_tags_file(input_fp, output_fp, tags_fp):
    with open(input_fp, 'r', encoding='utf-8') as infile, \
         open(output_fp, 'w', encoding='utf-8') as outfile, \
         open(tags_fp, 'w', encoding='utf-8') as tags_fh:
        for i, line in enumerate(infile):
            line = line.strip()
            line, tags = extract_tags(line)
            outfile.write(line + '\n')
            tags_fh.write(json.dumps(tags) + '\n')
    return output_fp, tags_fp

def reinsert_tags_file(input_fp, tags_fp, output_fp):
    with open(input_fp, 'r', encoding='utf-8') as infile, \
         open(tags_fp, 'r', encoding='utf-8') as tags_fh, \
         open(output_fp, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            tags = json.loads(tags_fh.readline().strip())
            if tags:
                line = reinsert_tags(line, tags)
            outfile.write(line + '\n')
    return output_fp

if __name__ == '__main__':
    import fire
    fire.Fire(reinsert_tags_file)
