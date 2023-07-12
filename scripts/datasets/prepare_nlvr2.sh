# Path: scripts/datasets/prepare_nlvr2.sh
#!/bin/sh
DATA_DIR=${1:-data}
for lang in "id" "sw" "ta" "tr" "zh-cn"
do
    for split in "train" "dev"
    do
        hatch run translate \
            --input-file $DATA_DIR/nlvr2r/nlvr2_${split}_en.json \
            --output-file $DATA_DIR/nlvr2r/nlvr2_${split}_${lang}.json \
            --lang $lang \
            --field "sentence"
    done
done
