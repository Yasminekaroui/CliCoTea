#!/bin/sh
# Purpose: Zero-shot evaluation of CLICOTEA on xFlickrCO
# Usage: sh scripts/zero-shot/zeroshot_retrieval.sh data lang flickr30k_image_root coco_image_root ckpt_path

DATA_DIR=${1:-"data"}
LANG=${2:-"de"}
TEST_FILE=$DATA_DIR/xFlickrCO/xFlickrCO_${LANG}_test.jsonl
FLICKR30K_IMAGE_ROOT=${3:-"flickr30k/images"}
COCO_IMAGE_ROOT=${4:-"coco/images"}
PATH_TO_CHECKPOINT=${5:-"experiments/flickr30k/albef_retrieval-flickr-$LANG/checkpoint-epoch-10.pth"}

hatch run zeroshot --task "retrieval" \
      --test-file $TEST_FILE \
      --flickr30k-image-root $FLICKR30K_IMAGE_ROOT \
      --coco-image-root $COCO_IMAGE_ROOT \
      --model-name "albef_retrieval" \
      --model-type "flickr" \
      --text-encoder-name "bert-base-multilingual-cased" \
      --text-encoder-ckpt $PATH_TO_CHECKPOINT \
      --device "cuda"
