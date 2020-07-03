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



usage="Preprocess a PAS 2015 Shared task corpus, in ACL 2019 style.\n\n

Arguments: \n
\n\t     -m  main directory where corpus lives
\n\t     -o  directory where output files will be put (default: 'output' within main directory)
"

while getopts "m:o:h" opt; do
    case $opt in
	h) echo -e $usage
	   exit
	   ;;
	m) maindir="$OPTARG"
	   ;;
	o) outputPath="$OPTARG"
	   ;;
       	\?) echo "Invalid option -$OPTARG" >&2
	    ;;
    esac
done

if [ "$maindir" = "" ]; then
    printf "No main directory given. Please use -m option.\n"
    exit 1
else
    printf "Processing files in main directory $maindir\n"
fi

if [ "$outputPath" = "" ]; then
    printf "No output directory given. Using default: 'output' folder inside main directory.\n"
    outputPath="$maindir/output"
fi
printf "Placing output in $outputPath\n"

mem=3G
alto="am-tools.jar"


train_and_dev="$maindir/en.pas.sdp"

test_id_input="$maindir/test/en.id.pas.tt"
test_id="$maindir/test/en.id.pas.sdp"

test_ood_input="$maindir/test/en.ood.pas.tt"
test_ood="$maindir/test/en.ood.pas.sdp"

tmp=$outputPath/tmp
mkdir -p $tmp


mkdir -p $outputPath

log=$outputPath/preprocessLog.txt
rm -f $log



train=$tmp/en.train.sdp
dev=$tmp/en.dev.sdp



if [ -f "$alto" ]; then
    printf "jar file found at $alto\n"
else
    printf "jar file not found at $alto, downloading it!\n"
    wget -O "$alto" http://www.coli.uni-saarland.de/projects/amparser/am-tools.jar
fi

# Split data into train/dev

java -cp $alto se.liu.ida.nlp.sdp.toolkit.tools.Splitter $train_and_dev $train $dev &> $log


# Decompose train set

mkdir -p $outputPath/train
java -Xmx$mem -cp $alto de.saar.coli.amrtagging.formalisms.sdp.pas.tools.CreateCorpusParallel -c $train -o "$outputPath/train" --prefix train &>> $log


# Decompose dev set
mkdir -p $outputPath/gold-dev
java -Xmx$mem -cp $alto de.saar.coli.amrtagging.formalisms.sdp.pas.tools.CreateCorpusParallel -c $dev -o "$outputPath/gold-dev" --prefix gold-dev --vocab "$outputPath/train/train-supertags.txt" &>> $log


# Prepare dev data (as if it was test data)
mkdir -p $outputPath/dev
java -Xmx$mem -cp $alto de.saar.coli.amrtagging.formalisms.sdp.tools.PrepareDevData -c $dev -o "$outputPath/dev" --prefix dev  --framework pas &>> $log
cp $dev "$outputPath/dev/dev.sdp"


#In-domain test:
mkdir -p $outputPath/test.id
java -Xmx$mem -cp $alto de.saar.coli.amrtagging.formalisms.sdp.tools.PrepareFinalTestData -c $test_id_input -o "$outputPath/test.id" --prefix test.id --framework pas &>> $log
cp $test_id "$outputPath/test.id/"

#out-of-domain test:
mkdir -p $outputPath/test.ood
java -Xmx$mem -cp $alto de.saar.coli.amrtagging.formalisms.sdp.tools.PrepareFinalTestData -c $test_ood_input -o "$outputPath/test.ood" --prefix test.ood --framework pas &>> $log
cp $test_ood "$outputPath/test.ood/"

#remove temp files
rm -f $train
rm -f $dev
rmdir $tmp

printf "process complete\n"

