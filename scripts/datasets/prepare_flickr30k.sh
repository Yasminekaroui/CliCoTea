# Path: scripts/datasets/prepare_flickr30k.sh
#!/bin/sh
DATA_DIR=${1:-data}
for lang in "de" "es" "id" "ru" "tr"
do
    for split in "train" "dev"
    do
        hatch run translate \
            --input-file $DATA_DIR/flickr30k/flickr30k_${split}_en.json \
            --output-file $DATA_DIR/flickr30k/flickr30k_${split}_${lang}.json \
            --lang $lang \
            --field "caption"
    done
done
