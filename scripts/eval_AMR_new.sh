#!/bin/bash
amcoll=$1
dir=$2 # output directory
ALTO_PATH=$(readlink -f "$3")

#Call like this:
#bash scripts/eval_AMR_new.sh <input.amconll> <outdir> <path to am-tools.jar>


# if these files are missing, run scripts/setup_AMR.sh
lookup="downloaded_models/lookup/lookupdata17/"
wordnet_path="downloaded_models/wordnet3.0/dict/"

echo "Evaluating to graphs and relabelling with threshold 10"

java -Xmx2G -cp "$ALTO_PATH" de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus -c "$1" -o "$dir" --relabel --wn "$wordnet_path" --lookup "$lookup" --th 10

