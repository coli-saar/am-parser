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

# TODO make this file safe for spaces in filenames

# default file paths
defaultmodel="downloaded_models/full_model.tar.gz"
mkdir -p downloaded_models
jar="am-tools.jar"

# Documenting parameters:
usage="Takes . \n\n

Required arguments: \n
\n\t     -i  input file:  Graph corpus in the original format. For example, the DM dev set in .sdp format.
\n\t\t     For EDS, make this the test.amr file that contains the gold graphs in PENMAN notation.
\n\t\t     For AMR, use the directory which contains all the test corpus files (e.g. data/amrs/split/test in the official AMR corpora)
\n\t     -o  output folder: where the results will be stored.
\n\t     -T  graph formalism of input file / that should be parsed to. Possible options: DM, PAS, PSD, AMR-2017 and EDS.

\noptions:

\n\t   -m  archived model file in .tar.gz format. If not given, the default model path $defaultmodel is used. If that
file does not exist, it will be downloaded automatically.
\n\t	 -f  faster, less accurate evaluation (flag; default false)
\n\t   -g  which gpu to use (its ID, i.e. 0, 1 or 2 etc). Default is -1, using CPU instead"

#defaults:
fast=false
gpu="-1"
# Gathering parameters:
while getopts "m:i:o:T:g:p:fh" opt; do
    case $opt in
	h) echo -e $usage
	   exit
	   ;;
	m) model="$OPTARG"
	   ;;
	i) input="$OPTARG"
	   ;;
	o) output="$OPTARG"
	   ;;
	T) type="$OPTARG"
	   ;;
	g) gpu="$OPTARG"
	   ;;
    p) pretrain="$OPTARG"
       ;;
	f) fast=true
	   ;;
	\?) echo "Invalid option -$OPTARG" >&2
	    ;;
    esac
done


if [ "$gpu" = "-1" ]; then
    echo "Warning: using CPU, this may be slow. Use -g to specify a GPU ID"
fi

if [ "$model" = "" ]; then
    model="$defaultmodel"
fi

if [ -f "$model" ]; then
    echo "model file found at $model"
else
    if [ "$model" = "$defaultmodel" ]; then
        echo "model not found at default model path. Downloading it!"
        wget -O "$defaultmodel" http://www.coli.uni-saarland.de/projects/amparser/full_model.tar.gz
    else
        echo "model not found at $model. Please check the -m parameter"
    fi
fi

if [ -f "$jar" ]; then
    echo "jar file found at $jar"
else
    echo "jar file not found at $jar, downloading it!"
    wget -O "$jar" http://www.coli.uni-saarland.de/projects/amparser/am-tools.jar
fi



if [ "$input" = "" ]; then
    printf "\n No input file given. Please use -i option.\n"
    exit 1
fi

if [ "$type" = "" ]; then
    printf "\n No graphbank type given. Please use -T option.\n"
    exit 1
elif [ "$type" = "AMR" ]; then
    type="AMR-2017" # to make it compatible with how we referred to this in the python code
fi

if [ "$output" = "" ]; then
    printf "\n No output folder path given. Please use -o option.\n"
    exit 1
fi

#Build fast_smatch
pushd external_eval_tools/fast_smatch
echo "Building fast_smatch (for evaluation of EDS)"
bash build.sh
popd

# Finished gathering parameters. We are now guaranteed to have the necessary arguments stored in the right place.
echo "Parsing input file $input with model $model to $type graphs, output in $output"

# create filename for amconll file
output=$output"/"
prefix=$type"_gold"
amconll_input=$output$prefix".amconll" # used as input for neural model, but we must create it first
amconll_prediction=$output$type"_pred.amconll" # where the neural model writes its prediction

# clean history output 
if [ -f "$amconll_prediction" ]; then
    rm $amconll_prediction
fi

# convert input file to AMConLL format
if [ "$type" = "DM" ] || [ "$type" = "PAS" ] || [ "$type" = "PSD" ]; then
    java -cp $jar de.saar.coli.amrtagging.formalisms.sdp.tools.PrepareFinalTestData -c $input -o $output -p $prefix --framework $type
elif [ "$type" = "EDS" ]; then
    java -cp $jar de.saar.coli.amrtagging.formalisms.eds.tools.PrepareTestData -c $input -o $output -p $prefix
elif [ "$type" = "AMR-2017" ]; then
    bash scripts/setup_AMR.sh
    bash scripts/preprocess_AMR_test.sh -i $input -o $output -j $jar
    #also creates goldAMR.txt in $output/
    mv $output/test.amconll $amconll_input
else
    echo "Graphbank type $type (-T option) not recognized! valid options are DM, PAS, PSD, EDS and AMR."
    exit 1
fi

# run neural net + fixed-tree decoder to obtain AMConLL file. Pass the --give_up option if we want things to run faster.
if [ "$fast" = "false" ]; then
    python3 parse_file_with_pretrained_model.py $model $type $amconll_input $amconll_prediction  -pretrain "$pretrain" --cuda-device $gpu 
else
    python3 parse_file_with_pretrained_model.py $model $type $amconll_input $amconll_prediction  -pretrain "$pretrain" --cuda-device $gpu --give_up 5 
fi

# convert AMConLL file (consisting of AM depenendcy trees) to final output file (containing graphs in the representation-specific format)
# and evaluate
echo "converting AMConLL to final output file .."
# TODO possibly clean up the if-then-else
if [ "$type" = "DM" ] || [ "$type" = "PAS" ]; then
    java -cp $jar de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus -c "$amconll_prediction" -o "$output$type" --gold "$input"
elif [ "$type" = "PSD" ]; then
    java -cp $jar de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPCorpus -c "$amconll_prediction" -o "$output$type" --gold "$input"
elif [ "$type" = "EDS" ]; then
    java -cp $jar de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus -c "$amconll_prediction" -o "$output$type"
    echo "EDM score:"
    python2 external_eval_tools/edm/eval_edm.py "$output$type".edm "$output$prefix"-gold.edm
    amrinput=${input%".edm"}".amr.txt"
    echo "Smatch score:"
    python2 external_eval_tools/fast_smatch/fast_smatch.py -f "$output$type".amr.txt "$output$prefix"-gold.amr.txt --pr
elif [ "$type" = "AMR-2017" ]; then
    bash scripts/eval_AMR_new.sh $amconll_prediction $output $jar
    bash scripts/smatch_AMR.sh $output

fi
