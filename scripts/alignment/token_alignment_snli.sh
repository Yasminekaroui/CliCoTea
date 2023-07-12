# Path: scripts/alignment/token_alignment_snli.sh
#!/bin/sh
DATA_DIR=${1:-data}
for lang in "ar" "es" "fr" "ru"
do
    for split in "train" "dev"
    do
        hatch run alignment \
            --src-file $DATA_DIR/snli/snli_${split}_en.json \
            --tgt-file $DATA_DIR/snli/snli_${split}_${lang}.json \
            --output-file $DATA_DIR/snli/word_pairs_${split}_en-${lang}.json \
            --field "hypothesis" \
            --device "cuda" \
            --model-name "aneuraz/awesome-align-with-co"
    done
done
