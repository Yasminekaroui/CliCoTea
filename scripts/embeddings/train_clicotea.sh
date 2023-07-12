#!/bin/sh
# Purpose: Train CLICOTEA
# Usage: sh scripts/embeddings/train_clicotea.sh [task]

TASK=${1:-"fickr30k"}
MODEL_NAME=${2:-"albef_retrieval"}
MODEL_TYPE=${3:-"flickr"}
LANG=${4:-"de"}

hatch run clicotea \
    --train-file "data/$TASK/word_pairs_train_en-$LANG.json" \
    --dev-file "data/$TASK/word_pairs_dev_en-$LANG.json" \
    --teacher-model-name $MODEL_NAME \
    --teacher-model-type $MODEL_TYPE \
    --student-model-name "bert-base-multilingual-cased" \
    --align-with-cls \
    --output-dir "experiments/$TASK/$MODEL_NAME-$MODEL_TYPE-$LANG" \
    --batch-size 128 \
    --epoch 50 \
    --lr 5e-5 \
    --device "cuda" \
    --exp-name "token-align" \
    --run-name "$MODEL_NAME-$MODEL_TYPE-$LANG" \
    --log-every-iters 100
