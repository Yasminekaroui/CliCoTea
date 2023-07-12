#!/bin/sh
# Purpose: Download datasets for training and testing
# Usage: sh scripts/zero-shot/download_datasets.sh [data_dir]

DATA_DIR=${1:-data}
mkdir -p $DATA_DIR

# download data for zero-shot cross-lingual transfer to downstream tasks
# data come from the IGLUE benchmark

# text-to-image/image-to-text retrieval
mkdir -p $DATA_DIR/xFlickrCO
wget -nc -O $DATA_DIR/xFlickrCO/xFlickrCO_de_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xFlickrCO/annotations/de/test.jsonl
wget -nc -O $DATA_DIR/xFlickrCO/xFlickrCO_es_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xFlickrCO/annotations/es/test.jsonl
wget -nc -O $DATA_DIR/xFlickrCO/xFlickrCO_id_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xFlickrCO/annotations/id/test.jsonl
wget -nc -O $DATA_DIR/xFlickrCO/xFlickrCO_ru_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xFlickrCO/annotations/ru/test.jsonl
wget -nc -O $DATA_DIR/xFlickrCO/xFlickrCO_tr_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xFlickrCO/annotations/tr/test.jsonl

# visual entailment
mkdir -p $DATA_DIR/xvnli
wget -nc -O $DATA_DIR/xvnli/xvnli_ar_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/XVNLI/annotations/ar/test.jsonl
wget -nc -O $DATA_DIR/xvnli/xvnli_es_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/XVNLI/annotations/es/test.jsonl
wget -nc -O $DATA_DIR/xvnli/xvnli_fr_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/XVNLI/annotations/fr/test.jsonl
wget -nc -O $DATA_DIR/xvnli/xvnli_ru_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/XVNLI/annotations/ru/test.jsonl

# visual reasoning
mkdir -p $DATA_DIR/marvl
wget -nc -O $DATA_DIR/marvl/marvl_id_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/marvl/zero_shot/annotations/marvl-id.jsonl
wget -nc -O $DATA_DIR/marvl/marvl_sw_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/marvl/zero_shot/annotations/marvl-sw.jsonl
wget -nc -O $DATA_DIR/marvl/marvl_ta_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/marvl/zero_shot/annotations/marvl-ta.jsonl
wget -nc -O $DATA_DIR/marvl/marvl_tr_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/marvl/zero_shot/annotations/marvl-tr.jsonl
wget -nc -O $DATA_DIR/marvl/marvl_zh_test.jsonl https://raw.githubusercontent.com/e-bug/iglue/main/datasets/marvl/zero_shot/annotations/marvl-zh.jsonl
