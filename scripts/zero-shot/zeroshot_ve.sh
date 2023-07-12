#!/bin/sh
# Purpose: Zero-shot evaluation of CLICOTEA on XVNLI
# Usage: sh scripts/zero-shot/zeroshot_ve.sh data lang flickr30k_image_root ckpt_path

DATA_DIR=${1:-"data"}
LANG=${2:-"fr"}
TEST_FILE=$DATA_DIR/xvnli/xvnli_${LANG}_test.jsonl
FLICKR30K_IMAGE_ROOT=${3:-"flickr30k/images"}
PATH_TO_CHECKPOINT=${4:-"experiments/snli/albef_classification-ve-$LANG/checkpoint-epoch-10.pth"}

hatch run zeroshot --task "ve" \
      --test-file $TEST_FILE \
      --flickr30k-image-root $FLICKR30K_IMAGE_ROOT \
      --model-name "albef_classification" \
      --model-type "ve" \
      --text-encoder-name "bert-base-multilingual-cased" \
      --text-encoder-ckpt $PATH_TO_CHECKPOINT \
      --device "cuda"
