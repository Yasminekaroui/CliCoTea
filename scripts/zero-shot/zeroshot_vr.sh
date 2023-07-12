#!/bin/sh
# Purpose: Zero-shot evaluation of CLICOTEA on XVNLI
# Usage: sh scripts/zero-shot/zeroshot_ve.sh data lang flickr30k_image_root ckpt_path

DATA_DIR=${1:-"data"}
LANG=${2:-"fr"}
TEST_FILE=$DATA_DIR/marvl/marvl_${LANG}_test.jsonl
MARVL_IMAGE_ROOT=${3:-"marvl-images"}
PATH_TO_CHECKPOINT=${4:-"experiments/nlvr2/albef_nlvr-nlvr-$LANG/checkpoint-epoch-10.pth"}

hatch run zeroshot --task "vr" \
      --test-file $TEST_FILE \
      --marvl-image-root $MARVL_IMAGE_ROOT/$LANG \
      --model-name "albef_nlvr" \
      --model-type "nlvr" \
      --text-encoder-name "bert-base-multilingual-cased" \
      --text-encoder-ckpt $PATH_TO_CHECKPOINT \
      --device "cuda"
