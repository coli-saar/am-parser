#!/bin/bash
##
## Copyright (c) 2020 Saarland University.
##
## This file is part of AM Parser
## (see https://github.com/coli-saar/am-parser/).
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##




usage="Preprocess the corpus.\n\n

Arguments: \n
\n\t     -i  directory in which the original AMR test files are, e.g. in *YOUR_PATH_TO_AMR_CORPUS*/data/amrs/split/test/
\n\t	 -m  memory limit to be used for the task (default: 6G)
\n\t	 -o  directory where output files will be put (default: new 'output'folder in main directory)
\n\t	 -j  path to am-tools jar file
\n\t	 -t  number of threads (default: 1)
"

while getopts "i:m:o:j:t:h" opt; do
    case $opt in
	h) echo -e $usage
	   exit
	   ;;
	i) inputdir="$OPTARG"
	   ;;
	m) memLimit="$OPTARG"
	   ;;
	o) outputPath="$OPTARG"
	   ;;
	j) jar="$OPTARG"
	   ;;
	t) threads="$OPTARG"
	   ;;
       	\?) echo "Invalid option -$OPTARG" >&2
	    ;;
    esac
done

if [ "$inputdir" = "" ]; then
    printf "\n No main directory given. Please use -i option."
    exit 1
else
    printf "Processing files in input directory $inputdir\n"
fi


if [ "$outputPath" = "" ]; then
    printf "\n No output directory given. Please use -o option.\n"
    exit 1
fi
printf "Placing output in $outputPath\n"


mkdir -p $outputPath
log=$outputPath/preprocess.log
if [ -f "$log" ]; then
    rm "$log"
fi

if [ "$jar" = "" ]; then
    printf "\n No jar file given. Please use -j option.\n"
    exit 1
else
    if [ -f "$jar" ]; then
        echo "jar file found at $jar"
    else
        echo "jar file not found at $jar, please check your -j option!"
        exit 1
    fi
fi

testNNdata=$outputPath/
testAltodata=$outputPath/alto/test/

# a lot of scripts live here so let's store it as a variable in case it changes
datascriptPrefix="de.saar.coli.amrtagging.formalisms.amr.tools.datascript"


# make subdirectories in the output directory
mkdir -p $testNNdata  # NN test set 
# for alto to read
mkdir -p $testAltodata             # test input for evaluation


NNdataCorpusName="namesDatesNumbers_AlsFixed_sorted.corpus"  # from which we get the NN training data
evalDataCorpusName="finalAlto.corpus"                        # from which we get the dev and test evaluation data
trainMinuteLimit=600                                         # limit for generating NN training data
devMinuteLimit=20                                            # limit for geneating NN dev data
if [ "$threads" = "" ]; then
    threads=1
fi
if [ "$memLimit" = "" ]; then
    memLimit=6G
fi
posTagger="downloaded_models/stanford/english-bidirectional-distsim.tagger"
nerTagger="downloaded_models/stanford/english.conll.4class.distsim.crf.ser.gz"
wordnet="downloaded_models/wordnet3.0/dict/"

# disable use of conceptnet by replacing this with 'conceptnet=""'
#conceptnet="--conceptnet resources/conceptnet-assertions-5.7.0.csv.gz"
conceptnet=""


# Raw dev and test to Alto format

# test set
testRawCMD="java -Xmx$memLimit -cp $jar $datascriptPrefix.FullProcess --amrcorpus $inputdir --output $testAltodata  >>$log 2>&1"
printf "\nconverting test set to Alto format for evaluation\n"
printf "\nconverting test set to Alto format for evaluation\n" >> $log
echo $testRawCMD >> $log
eval $testRawCMD

# test eval input data preprocessing
testEvalCMD="java -Xmx$memLimit -cp $jar $datascriptPrefix.MakeDevData -c $testAltodata -o $testNNdata --stanford-ner-model $nerTagger --tagger-model $posTagger >>$log 2>&1"
printf "\ngenerating evaluation input (full corpus) for test data\n"
printf "\ngenerating evaluation input (full corpus) for test data\n" >> $log
echo $testEvalCMD  >> $log
eval $testEvalCMD

# move some stuff around
printf "\nMoving some files around. If you get errors here, something above didn't work. Check the logfile in $log\n"
cp $testAltodata/raw.amr $testNNdata/goldAMR.txt


#Create empty amconll for test set
emptyTestAmconllCMD="java -Xmx$memLimit -cp $jar de.saar.coli.amrtagging.formalisms.amr.tools.PrepareTestDataFromFiles -c $testNNdata -o $outputPath --prefix test --stanford-ner-model $nerTagger >>$log 2>&1"
printf "\nGenerate empty amconll test data\n"
eval $emptyTestAmconllCMD

printf "\neverything is in $outputPath\n"
