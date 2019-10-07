#!/bin/bash

# download Stanford NER and POS taggers. They are licensed under the GPL license, full details here:
# https://nlp.stanford.edu/software/tagger.shtml
# https://nlp.stanford.edu/software/CRF-NER.html
ner_target="downloaded_models/stanford/english.conll.4class.distsim.crf.ser.gz"
pos_target="downloaded_models/stanford/english-bidirectional-distsim.tagger"

mkdir tmp
mkdir -p downloaded_models/stanford # the p stands for parents, i.e. parent folders are also created if necessary
# NER
if [ -f "$ner_target" ]; then
    echo "NER model found at $ner_target"
else
    echo "NER model not found at $ner_target, downloading it"
    wget https://nlp.stanford.edu/software/stanford-ner-2018-10-16.zip -O tmp/stanford-ner-2018-10-16.zip
    unzip tmp/stanford-ner-2018-10-16.zip -d tmp/
    mv tmp/stanford-ner-2018-10-16/classifiers/english.conll.4class.distsim.crf.ser.gz $ner_target
fi
# POS
if [ -f "$pos_target" ]; then
    echo "POS model found at $pos_target"
else
    echo "POS model not found at $pos_target, downloading it"
    wget https://nlp.stanford.edu/software/stanford-postagger-2018-10-16.zip -O tmp/stanford-postagger-2018-10-16.zip
    unzip tmp/stanford-postagger-2018-10-16.zip -d tmp/
    mv tmp/stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger $pos_target
fi


# wordnet
lookup_check="downloaded_models/wordnet3.0/dict/data.noun" # just check one file, should all be there or none
if [ -f "$wordnet_check" ]; then
    echo "WordNet found in downloaded_models/wordnet3.0 (If there are problems with missing files in that folder, delete it and run this script again.)"
else
    wget http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz -O tmp/WordNet-3.0.tar.gz
    tar -xzvf tmp/WordNet-3.0.tar.gz -C tmp
    mkdir -p downloaded_models/wordnet3.0
    mv tmp/WordNet-3.0/dict downloaded_models/wordnet3.0/
fi
rm -rf tmp

# our own lookup files
lf="downloaded_models/lookup" # lookup folder
lookup_check="$lf/lookupdata15/nameLookup.txt" # just check one file, should all be there or none
mkdir -p downloaded_models/lookup # the p stands for parents, i.e. parent folders are also created if necessary
if [ -f "$lookup_check" ]; then
    echo "Lookup data found at $lf. (If there are problems with missing files in that folder, delete it and run this script again.)"
else
    # 2015 dataset
    wget https://coli-saar-data.s3.eu-central-1.amazonaws.com/lookupdata15.zip -O $lf/lookupdata15.zip
    unzip $lf/lookupdata15.zip -d $lf
    rm $lf/lookupdata15.zip
    # 2017 dataset
    wget https://coli-saar-data.s3.eu-central-1.amazonaws.com/lookupdata17.zip -O $lf/lookupdata17.zip
    unzip $lf/lookupdata17.zip -d $lf
    rm $lf/lookupdata17.zip
fi