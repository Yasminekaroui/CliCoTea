# Path: scripts/datasets/prepare_nlvr2.sh
#!/bin/sh
DATA_DIR=${1:-data}
for lang in "ar" "es" "fr" "ru"
do
    for split in "train" "dev"
    do
        hatch run translate \
            --input-file $DATA_DIR/snli/snli_${split}_en.json \
            --output-file $DATA_DIR/snli/snli_${split}_${lang}.json \
            --lang $lang \
            --field "hypothesis" "premise"
    done
done
