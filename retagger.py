#!/usr/bin/env python3.7
# -*- coding: utf-8 -*
"""
"""
import json
import logging
import math
import os
import pathlib
import random
import re
import shutil
import string
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
# URL regex source (Gruber):
# https://gist.github.com/gruber/8891611
URL_REGEX = r"""(?:(?i)[^\w](?:(?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))[^\w])"""
URL_TEMPL = ' [URL{}] '
#
# EMAIL regex source (Regular-Expressions.info) 2nd to last one: 
# http://www.regular-expressions.info/email.html
EMAIL_REGEX = r"(?:[^\w][a-z0-9!#$%&'*+/=?^_‘{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_‘{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?[^\w])"
EMAIL_TEMPL = ' [EML{}] '
#
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
            tags = json.loads(tags_fh.readline().strip())
            if tags:
                line, tags = reinsert_tags(line, tags)
            outfile.write(line + '\n')
    return output_fp
