#/bin/bash
# Call with data/AMR/2015/

echo Preparing lexicon for $1
mkdir -p "$1/lexicon"
cat "$1/train/train.amconll" "$1/gold-dev/gold-dev.amconll" | grep -vE "^#" | cut -f11 | sort | uniq | grep . > "$1/lexicon/edges.txt"
cat "$1/train/train.amconll" "$1/gold-dev/gold-dev.amconll" | grep -vE "^#" | cut -f8 | sort | uniq | grep . > "$1/lexicon/lex_labels.txt"


python topdown_parser/tools/prepare_constants.py "$1/lexicon/" --corpora "$1/train/train.amconll" "$1/gold-dev/gold-dev.amconll"


