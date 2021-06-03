"""
Search for .tmx files in an input dir and extract parallel (moses format)
text files for language pairs. The file should have xx-yy in its name, where
xx and yy are languages from the list of langs.
"""
import os
from itertools import permutations

import xml.etree.ElementTree

import extractors
import utils

logger = utils.setup_logger('extract_all_tmxs')

def extract_tmxs(indir, outdir, langs, skip_tmx_parse_errors=True):
    extracted = []
    skipped = []
    directions = list(permutations(langs, 2))
    for root, dirs, files in os.walk(indir):
        for filename in files:
            base, ext = os.path.splitext(filename)
            if ext != '.tmx':
                continue

            done = False
            fp = os.path.join(root, filename)
            relative_name = os.path.relpath(fp, indir)
            this_outdir = os.path.dirname(relative_name)
            if this_outdir:
                os.makedirs(this_outdir, exist_ok=True)

            for direction in directions:
                direction_str = '-'.join(direction)
                if direction_str in fp:
                    outname = os.path.join(outdir, this_outdir, filename)
                    if (
                        os.path.exists(outname + f'.{direction[0]}') and
                        os.path.exists(outname + f'.{direction[1]}') 
                    ):
                        logger.info(f"Skipping existing corpus: {outname}")
                        continue
                    logger.info(f"Extracting {fp}")

                    try:
                        extractors.extract_tmx(fp, outname, direction[0], direction[1])
                    except xml.etree.ElementTree.ParseError as e:
                        logger.warning(f'ERROR PARSING {fp}: {e}')
                        if not skip_tmx_parse_errors:
                            raise
                        else:
                            continue
                    
                    extracted.append(fp)
                    done = True
            if not done:
                skipped.append(fp)

    print(f'Extracted: {extracted}')
    print(f'Skipped: {skipped}')

if __name__ == '__main__':
    import fire
    fire.Fire(extract_tmxs)