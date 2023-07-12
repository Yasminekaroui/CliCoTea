# CLiCoTEA: Cross-Lingual Contextualised Token Embedding Alignment

This code reproduces the results from ACL 2023 paper ["Stop Pre-Training: Adapt Visual-Language Models to Unseen Languages"](https://arxiv.org/abs/2306.16774).

## Installation
These dependencies must be installed:
- [Hatch](https://hatch.pypa.io/): for managing the Python package
- [gdown](https://pypi.org/project/gdown/): for downloading the datasets
```
pip install hatch gdown
```

## Prepare datasets

Download all datasets for training the Cross-Lingual Contextualised Token Embedding Alignment and the Zero-Shot Cross-Lingual transfer to downstream tasks:
```bash
bash scripts/datasets/download_datasets.sh data
```
The [archive](https://drive.google.com/file/d/1qG3smKmz6l9BDwTxOwjcWhT4IGh4jIfB/view?usp=sharing) contains the original files from [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/), [SNLI](https://huggingface.co/datasets/snli) and [NLVR2](https://github.com/lil-lab/nlvr/tree/master/nlvr2) which are all in English.
It also includes the translated files for each language required in the downstream tasks.

Note that the translation of train/dev sets of Flickr30k, SNLI and NVLR2 datasets has be done using [Googletrans](https://py-googletrans.readthedocs.io/en/latest/) package, running the following commands:
```bash
bash scripts/datasets/prepare_flickr30k.sh
bash scripts/datasets/prepare_snli.sh
bash scripts/datasets/prepare_nlvr2.sh
```

## Compute token alignment with `awesome-align` model

```bash
bash scripts/alignment/token_alignment_flickr30k.sh
bash scripts/alignment/token_alignment_snli.sh
bash scripts/alignment/token_alignment_nlvr2.sh
```
This should create aligned word pairs in `data` folder for each dataset as follows:
```
data/
   flickr30k/
      word_pairs_dev_en-de.json
      word_pairs_dev_en-es.json
      word_pairs_dev_en-id.json
      word_pairs_dev_en-ru.json
      word_pairs_dev_en-tr.json
      word_pairs_train_en-de.json
      word_pairs_train_en-es.json
      word_pairs_train_en-id.json
      word_pairs_train_en-ru.json
      word_pairs_train_en-tr.json
   nlvr2/
      word_pairs_dev_en-id.json
      word_pairs_dev_en-sw.json
      word_pairs_dev_en-ta.json
      word_pairs_dev_en-tr.json
      word_pairs_dev_en-zh-cn.json
      word_pairs_train_en-id.json
      word_pairs_train_en-sw.json
      word_pairs_train_en-ta.json
      word_pairs_train_en-tr.json
      word_pairs_train_en-zh-cn.json
   snli/
      word_pairs_dev_en-ar.json
      word_pairs_dev_en-es.json
      word_pairs_dev_en-fr.json
      word_pairs_dev_en-ru.json
      word_pairs_train_en-ar.json
      word_pairs_train_en-es.json
      word_pairs_train_en-fr.json
      word_pairs_train_en-ru.json
```

## Train CLiCoTEA

Train CLiCoTEA by running the following commands (default options should be modified from the bash script):
```bash
# train CLiCoTEA for image/text retrieval on flickr30k in German
bash scripts/embeddings/train_clicotea.sh flickr30k albef_retrieval flickr de

# train CLiCoTEA for visual reasoning on NLVR2 in Swahili
bash scripts/embeddings/train_clicotea.sh nlvr2 albef_nlvr nlvr sw

# train CLiCoTEA for visual entailment on SNLI in French
1bash scripts/embeddings/train_clicotea.sh snli albef_classification ve fr
```
Note that we start from pre-trained ALBEF models which are available in [LAVIS](https://github.com/salesforce/LAVIS) package.

## Zero-shot transfer to unseen languages

1. Download the images of the downstream tasks from the official website:
- [Flickr30K](https://www.kaggle.com/hsankesara/flickr-image-dataset)
- [MS COCO](https://cocodataset.org/#download)
- [MaRVL](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/42VZ4P)

Text data can be downloaded from the [IGLUE](https://github.com/e-bug/iglue) Benchmark with:
```bash
bash scripts/zero-shot/download_datasets.sh
```

2. Run zero-shot evaluation:
```bash
DATA_DIR="<path to folder containing test files>"
LANG="<language to test>"
FLICKR30K_IMAGE_ROOT="<place path to image folder>"
COCO_IMAGE_ROOT="<place path to image folder>"
MARVL_IMAGE_ROOT="<place path to image folder>"
PATH_TO_CHECKPOINT="<place path to model checkpoint>"
```
- Retrieval task on xFlickrCO

```bash
bash scripts/zero-shot/zeroshot_retrieval.sh $DATA_DIR $LANG $FLICKR30K_IMAGE_ROOT $COCO_IMAGE_ROOT $PATH_TO_CHECKPOINT
```

- Visual entailment task on XVNLI

```bash
bash scripts/zero-shot/zeroshot_ve.sh $DATA_DIR $LANG $FLICKR30K_IMAGE_ROOT $PATH_TO_CHECKPOINT
```

- Visual reasoning task on MaRVL

```bash
bash scripts/zero-shot/zeroshot_vr.sh $DATA_DIR $LANG $MARVL_IMAGE_ROOT $PATH_TO_CHECKPOINT
```

## Running tests

Running all tests:
```bash
hatch run test:run
```

Or running a specific test:
```bash
hatch run test:run -k test_get_token_pairs
```

## Citation

Please cite as:

```bibtex
@inproceedings{clicotea,
   title = "Stop Pre-Training: Adapt Visual-Language Models to Unseen Languages",
    author = "Karoui, Yasmine  and
      Lebret, R{\'e}mi  and
      Foroutan Eghlidi, Negar  and
      Aberer, Karl",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.32",
    pages = "366--375",
}
```
