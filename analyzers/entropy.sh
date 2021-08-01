#! /bin/bash

# help if # args != 3
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 jar neural EM"
    echo "   * jar: path to am-tools.jar file"
    echo "   * neural: path to neural amconll files' prefix. i.e. the parent of DM/ PSD/ PAS/ and AMR/ for neural output"
    echo "   * EM: path to EM amconll files' prefix"
    echo " Example: bash entropy.sh ~/am-tools/build/libs/am-tools.jar ~/amconll_files/neural/ ~/amconll_files/EM/"
    exit 0
fi


jar_path=$1  # path to am-tools.jar
path_neural=$2 # path to neural folder
path_EM=$3  # path to EM amconll folder


for corpus in DM PSD PAS AMR
do
    java -Xmx2G -cp $jar_path de.saar.coli.amtools.decomposition.SupertagEntropy $path_neural $path_EM $corpus
done

