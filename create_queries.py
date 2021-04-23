import json
import logging
import math
import os
import random
import shutil
from functools import partial
from multiprocessing import cpu_count

from parallely import pll_multi

logger = logging.getLogger('clean')
logger.setLevel(logging.INFO)

CPU_COUNT = max(1, math.floor(cpu_count() / 2) - 1)
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
STOPWORDS = f'{ROOT_DIR}/stopwords-json/stopwords-all.json'

#Characters to strip from the query terms, so we don't have punct in query
#NOTE: don't strip dashes, underscores, quotes, apostrophe
TOSTRIP=r'?/.,<>;:\|][{}=+)(*&^%$#@!`~'
#Avoid adding a query to a line this percentage of the time (use this to
#ensure that the model doesn't forget how to translate without queries)
DROPOUT=0.2
#These are the n-grams that a query could have
NGRAMS=[1,2,3]

def _extract_random_span(
        text: str,
        tgt_lang,
        tgt_stopwords=None,
        to_strip=TOSTRIP,
        dropout=DROPOUT,
        ngrams=NGRAMS,
    ) -> str:
    r"""Extract a random span from the target language text."""
    #TODO: implement some "content word" method of finding query terms for
    #training data (tf-idf or something?) rather than just non-stopwords?

    if ngrams and random.uniform(0, 1) > dropout:
        text = text.strip().split()
        n_idx = random.randrange(len(ngrams))
        n = ngrams[n_idx]
        N = len(text) - n
        if N < 0:
            span = ''
        elif N == 0:
            span = text[0:n]
            span = ' '.join(span)
        else:
            span = None
            for i in range(N):
                start = random.choice(range(N))
                span = text[start:start+n]
                if not tgt_lang or not tgt_stopwords:
                    break
                span = [s.translate(str.maketrans('', '', to_strip)) 
                        for s in span]
                if span[-1].strip().lower() not in tgt_stopwords:
                    break
            if span is None:
                span = ''
            else:
                span = ' '.join(span).strip()
    else:
        span = ''
    return span

def _create_queries(
        input_fps: list, 
        output_dir: str,
        tgt_lang,
        tgt_stopwords=None,
        to_strip=TOSTRIP,
        dropout=DROPOUT,
        ngrams=NGRAMS,
    ) -> str:
    name = os.path.basename(input_fps[0])
    queries_fp = os.path.join(output_dir, name) + '.queries'

    with open(input_fps[1], 'r', encoding='utf-8') as tgt_fh, \
         open(queries_fp, 'w', encoding='utf-8') as queries_fh:
        for tgt_line in tgt_fh:
            tgt_line = tgt_line.strip()

            tgt_span = _extract_random_span(
                tgt_line,
                tgt_lang,
                tgt_stopwords,
                to_strip,
                dropout,
                ngrams
            )
            queries_fh.write(tgt_span + os.linesep)

    return [queries_fp] #return a list for use with pll_multi

def create_queries(
        src: str, tgt: str, 
        output_dir: str, 
        src_lang: str, tgt_lang: str, 
        to_strip=TOSTRIP,
        dropout=DROPOUT,
        ngrams=NGRAMS,
        **kwargs
    ) -> dict:
    name = os.path.basename(src)
    queries_fp = os.path.join(output_dir, name + '.queries')

    #avoid overwriting/redoing work
    if os.path.exists(queries_fp):
        logger.info(f"Skipping; files already exist: {queries_fp}")
        #abuse the src/tgt interface so we can use this output in future steps
        return {'src': queries_fp, 'tgt': ''}

    tgt_stopwords = {}
    with open(STOPWORDS, 'r', encoding='utf-8') as infile:
        all_stopwords = json.load(infile)
        if tgt_lang in all_stopwords:
            tgt_stopwords = all_stopwords[tgt_lang]

    #parallel process creating the queries file 
    part = partial(
        _create_queries,
        output_dir=output_dir,
        tgt_lang=tgt_lang,
        tgt_stopwords=tgt_stopwords,
        to_strip=to_strip,
        dropout=dropout,
        ngrams=ngrams,
    )
    queries_out_fps = pll_multi(
        [src, tgt], 
        part, 
        n_jobs=CPU_COUNT, 
        outdir=output_dir, 
        output_name=name
    )

    #rename the file coming from pll_multi to the appropriate name
    shutil.move(queries_out_fps[0], queries_fp)

    #abuse the src/tgt interface so we can use this output in future steps
    return {'src': queries_fp, 'tgt': ''}
