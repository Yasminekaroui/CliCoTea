# Path: scripts/alignment/token_alignment_flickr30k.sh
#!/bin/sh
DATA_DIR=${1:-data}
for lang in "de" "es" "id" "ru" "tr"
do
    for split in "train" "dev"
    do
        hatch run alignment \
            --src-file $DATA_DIR/flickr30k/flickr30k_${split}_en.json \
            --tgt-file $DATA_DIR/flickr30k/flickr30k_${split}_${lang}.json \
            --output-file $DATA_DIR/flickr30k/word_pairs_${split}_en-${lang}.json \
            --field "caption" \
            --device "cuda" \
            --model-name "aneuraz/awesome-align-with-co"
    done
done
