# Path: scripts/alignment/token_alignment_nlvr2.sh
#!/bin/sh
DATA_DIR=${1:-data}
for lang in "id" "sw" "ta" "tr" "zh-cn"
do
    for split in "train" "dev"
    do
        hatch run alignment \
            --src-file $DATA_DIR/nlvr2/nlvr2_${split}_en.json \
            --tgt-file $DATA_DIR/nlvr2/nlvr2_${split}_${lang}.json \
            --output-file $DATA_DIR/nlvr2/word_pairs_${split}_en-${lang}.json \
            --field sentence \
            --device "cuda" \
            --model-name "aneuraz/awesome-align-with-co"
    done
done
