#!/bin/bash

dir=$1
ALTO_PATH=$(readlink -f "$2")

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")


# call in 2019rerun with bash scripts/eval.sh
# needs in $dir: unlabeled graphs parserOut.txt, their indices in the original corpus indices.txt, and best non-null labels per token labels.txt (the latter can be taken from tagger_out)


ln -s "$SCRIPTPATH/../test/goldAMR.txt" "$dir/goldAMR.txt"
ln -s "$SCRIPTPATH/../test/literal.txt" "$dir/literal.txt"
ln -s "$SCRIPTPATH/../test/sentences.txt" "$dir/sentences.txt"
ln -s "$SCRIPTPATH/../test/pos.txt" "$dir/pos.txt"
ln -s "$SCRIPTPATH/../metadata/wikiLookup.txt" "$dir/wikiLookup.txt"
ln -s "$SCRIPTPATH/../metadata/nameLookup.txt" "$dir/nameLookup.txt"
ln -s "$SCRIPTPATH/../metadata/nameTypeLookup.txt" "$dir/nameTypeLookup.txt"
ln -s "$SCRIPTPATH/../metadata/words2labelsLookup.txt" "$dir/words2labelsLookup.txt"
# ln -s "$SCRIPTPATH/../tagger_out/labels.txt" "$dir/labels.txt"

echo "relabelling with threshold 10"

# run the relabeller
java -Xmx2G -cp "$ALTO_PATH" de.saar.coli.amrtagging.formalisms.amr.tools.Relabel $dir/ "$SCRIPTPATH/../metadata/" "$SCRIPTPATH/../metadata/wordnet/3.0/dict/" 10

echo "final post-processing"

# now we have a relabelled.txt in this directory
# do more post-processing
sed -E 's/\(u_[0-9]+ \/ ([-+0-9]+)\)/\1/g' $dir/relabeled.txt | sed -E 's/\(explicitanon[0-9]+ \/ ([^"()]+)\)/"\1"/g' | sed -E 's/\(explicitanon[0-9]+ \/ ("[^"]+")\)/\1/g' | sed -E 's/"([-+0-9]+)"/\1/g' > $dir/relabeled_fixed.txt

# compute smatch score
echo "smatching in $dir"

python "$SCRIPTPATH/../smatch/smatch.py" -f $dir/relabeled_fixed.txt $dir/gold_orderedAsRelabeled.txt --significant 4 --pr | tee $dir/smatch.txt
