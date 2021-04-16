import io
import os
import requests
import zipfile 

from tqdm import tqdm

def query_opus_corpora(src_lang, tgt_lang):
    opus_url = f"http://opus.nlpl.eu/opusapi/?source={src_lang}&target={tgt_lang}&preprocessing=moses"
    try:
        data = requests.get(opus_url).json()
        corpora = {}
        for corpus in data["corpora"]:
            if corpus["corpus"] not in corpora:
                corpora[corpus["corpus"]] = corpus
    except Exception as e:
        logger.info(f"Skipping {src_lang}-{tgt_lang} due to error: {e}.")
        corpora = {}
    print(corpora)
    return corpora

def download_opus_corpora(corpora, root_dir):
    downloaded = []
    for corpus_name in tqdm(corpora):

        output_dir = os.path.join(root_dir, corpus_name)
        corpus_url = corpora[corpus_name]['url']

        os.makedirs(output_dir, exist_ok=True)

        r = requests.get(corpus_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(output_dir)

        downloaded.append(output_dir)

    return downloaded

def main(src_lang, tgt_lang, outdir):
    corpora = query_opus_corpora(src_lang, tgt_lang)
    download_opus_corpora(corpora, outdir)


if __name__ == '__main__':
    import fire
    fire.Fire(main)