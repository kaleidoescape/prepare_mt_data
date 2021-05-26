# prepare_mt_data

A loose collection of scripts for preparing parallel machine translation data.

# Requirements

```
pip install -r requirements.txt
```

# Usage

## Config file preparation

Many of these scripts read a yaml file that contains parallel dataset information. For example, a `config.data.yml` file may look like this:

```
data:
  da2de_TED2020:
    src: path/to/data/TED2020/TED2020.da-de.da
    src_lang: da
    tgt: path/to/data/TED2020/TED2020.da-de.de
    tgt_lang: de
```

A yaml file can be auto-generated from a directory of datasets with the same prefix, e.g. TED2020.da-de.da, TED2020.da-de.en, Paracrawl.en-sv.en, Paracrawl.en-sv.sv, etc. using the following script, which will search the directory recursively to find datasets with the requested languages: 

```
python scripts/collect_data_files.py \
    path/to/data \
    config.data.yml \
    "['is', 'nb', 'sv', 'da', 'en', 'de']"
```

The `prepare_training_data.py` script can use the resulting config with different commands to perform cleaning steps (see below), and write out the newly cleaned filepaths to a new config that can be used in subsequent steps.

Multiple configs can be used (later configs override earlier configs). This can be taken advantage of for those steps that may require additional arguments. For example, the language identification step (see below) can use language id aliases as an argument, which can be placed in a separate config for more flexibility. 

## Cleaning Steps

The charfix command removes characters that affect line counting when using certain linux tools (e.g. extra `\r` coming from Windows line breaks). Without this, parallel sentences can become mis-aligned later on in the process, so this is an important step:

```
python ./scripts/prepare_training_data.py \
    --configs config.data.yml \
    --outdir data/prepared/ \
    --command charfix \
    --name data
```

The bifix step uses Bifixer heuristics to:
- fix orthographic encoding errors such as mojibake (see `bifixclean/restorative_cleaning.py` for specifics)
- remove sentences with empty source/target
- ignore sentences over 5000 tokens in length
- remove duplicate sentences

```
python ./scripts/prepare_training_data.py \
    --configs data/prepared/config.charfix.data.yml \
    --outdir data/prepared/ \
    --command bifix \
    --name data
```

The biclean step uses Bicleaner 'hardrules' heuristics to filter out sentences with too many non-target language characters in them (see `bifixclean/bicleaner_hardrules.py` for specifics):

```
python ./scripts/prepare_training_data.py \
    --configs data/prepared/config.bifix.data.yml \
    --outdir data/prepared \
    --command biclean \
    --name data
```

The dedup step deduplicates identical translation pairs (i.e. both source and target are the same; if just one side is different, it will be retained). WARNING! This step uses `sort -u` under the hood, so the output data files will come out sorted.

```
python ./scripts/prepare_training_data.py \
    --configs data/prepared/config.biclean.data.yml \
    --outdir data/prepared \
    --command dedup \
    --name data
```

The langcheck step uses fastText to extract only the target languages from the datasets. It can take additional arguments for language id aliases. For example, the following `config.langid_aliases.yml` allows the `nb` language to be identified as either `nb` or `no` (this is necessary for `nb` because fastText's model only has `no` and it will delete too many correct sentences otherwise):

```
args:
  aliases:
    nb:
      - "no"
      - "nb"
```

```
python ./scripts/prepare_training_data.py \
    --configs data/prepared/config.dedup.data.yml config.langid_aliases.yml \
    --outdir data/prepared \
    --command langcheck \
    --name data
```

The tagprotect step replaces emails, urls, and xml tags with special tokens. It creates parallel `.repls` files that contain a json list of replacements for each each line, so that the replacements can later be re-inserted:

```
python ./scripts/prepare_training_data.py \
    --configs data/prepared/config.langcheck.data.yml \
    --outdir data/prepared \
    --command tagprotect \
    --name data
```

# Acknowledgements


We use one part of the [Bifixer](https://github.com/bitextor/bicleaner) and [Bifixer](https://github.com/bitextor/bifixer) cleaning tools (the heuristics only) inside our code for cleaning parallel sentences. 

```
@InProceedings{prompsit:2020:EAMT,
  author    = {Gema Ram\'{i}rez-S\'{a}nchez and Jaume Zaragoza-Bernabeu and Marta Ba{\~n}\'{o}n and Sergio Ortiz-Rojas},
  title     = {Bifixer and Bicleaner: two open-source tools to clean your parallel data.},
  booktitle = {Proceedings of the 22nd Annual Conference of the European Association for Machine Translation},
  pages	    = {291--298},
  isbn      = {978-989-33-0589-8},
  year	    = {2020},
  month     = {November},
  address   = {Lisboa, Portugal},
  publisher = {European Association for Machine Translation}
}
```

We use [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) pre-trained word vectors for language identification.

```
@inproceedings{grave2018learning,
  title={Learning Word Vectors for 157 Languages},
  author={Grave, Edouard and Bojanowski, Piotr and Gupta, Prakhar and Joulin, Armand and Mikolov, Tomas},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```
