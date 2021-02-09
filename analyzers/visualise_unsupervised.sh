#! /bin/bash

# help shown if no arguments given
if [ $# == 0 ]; then
    echo "Usage: $0 file_path"
    echo "   * file_path: path to amconll file to sample from."
    echo " Example: bash visualise_unsupervised.sh ~/unsupervized/analysis/neural/AMR/APP_S0.amconll"
    exit 0
fi



path=$1  # path to amconll file

python ./compare_amconll.py $path $path
