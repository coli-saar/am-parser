#! /bin/bash

# Writes all counts in de.saar.coli.amtools.decomposition.analysis Count scripts for all 4 corpus types. Assumes amconll files are called AMR.amconll etc.

# help if # args != 3
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 jar input_path output_path"
    echo "   * jar: path to am-tools.jar file"
    echo "   * input_path: path to neural amconll files' prefix. i.e. the parent of DM/ PSD/ PAS/ and AMR/"
    echo "   * output_path: path to where you want to write the analysis"
    echo " Example: bash all_counts.sh ~/am-tools/build/libs/am-tools.jar ~/amconll_files/neural/ ~/unsupervized/analysis/neural/"
    exit 0
fi



jar_path=$1 # path to the am-tools jar file
corpus_path=$2 # path to the directory where you have all the corpora
output_path=$3 # path to the directory to write the analysis


for corpus in DM PSD PAS AMR
do
    for count in Sources Edges Supertags  # this works because the scripts are called e.g. CountSources.java
    do
	java -Xmx2G -cp $jar_path de.saar.coli.amtools.decomposition.analysis.Count$count "$corpus_path" "$output_path" "$corpus"
    done
done

