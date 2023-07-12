#!/bin/sh
# Purpose: Download datasets for training CLiCoTEA
# Usage: sh scripts/datasets/download_datasets.sh [data_dir]

DATA_DIR=${1:-data}
mkdir -p $DATA_DIR
# download data for cross-lingual contextualized token embeddings alignment
gdown "https://drive.google.com/uc?id=1uXk_r3fJ1-DdyA9VdrGE5YGZloNQqprS" -O $DATA_DIR/data.zip
unzip $DATA_DIR/data.zip -d $DATA_DIR
rm $DATA_DIR/data.zip
