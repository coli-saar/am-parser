#!/bin/bash

# default file paths
defaultmodel="example/raw_text_model.tar.gz"
jar="am-tools-all.jar"

# Documenting parameters:
usage="Takes . \n\n

Required arguments: \n
\n\t     -i  input file: text file with one sentence per line. Assumes the file is already tokenized with spaces as delimiters.
\n\t     -o  output folder: where the results will be stored.
\n\t     -T  desired type of output formalism. Possible options: DM, PAS, PSD, EDS (this raw text version does not support AMR).

\noptions:

\n\t   -m  archived model file in .tar.gz format. If not given, the default model path $defaultmodel is used. If that
file does not exist, it will be downloaded automatically (NOT YET IMPLEMENTED)
\n\t	 -f  faster, less accurate evaluation (flag; default false)
\n\t   -g  which gpu to use (its ID, i.e. 0, 1 or 2 etc). Default is -1, using CPU instead"

#defaults:
fast=false
gpu="-1"
# Gathering parameters:
while getopts "m:i:o:T:g:f" opt; do
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
	f) fast=true
	   ;;
	\?) echo "Invalid option -$OPTARG" >&2
	    ;;
    esac
done


if [ "$model" = "" ]; then
    model="$defaultmodel"
fi

if [ -f "$model" ]; then
    echo "model file found at $model"
else
    if [ "$model" = "$defaultmodel" ]; then
        echo "model not found at default model path. Downloading it!"
        # TODO replace this with code that downloads the model from the internet
        cp /local/mlinde/am-parser/models/mtl_bert_minimum/model.tar.gz "$defaultmodel"
    else
        echo "model not found at $model. Please check the -m parameter"
    fi
fi

if [ -f "$jar" ]; then
    echo "jar file found at $jar"
else
    echo "jar file not found at $jar, downloading it!"
    # TODO replace this with code that downloads the jar file from the internet
    cp /proj/irtg.shadow/tools/am-tools-all.jar "$jar"
fi



if [ "$input" = "" ]; then
    printf "\n No input file given. Please use -i option.\n"
    exit 1
fi

if [ "$type" = "" ]; then
    printf "\n No output graphbank type given. Please use -T option.\n"
    exit 1
fi

if [ "$output" = "" ]; then
    printf "\n No output file path. Please use -o option.\n"
    exit 1
fi
# Finished gathering parameters. We are now guaranteed to have the necessary arguments stored in the right place.
echo "Parsing raw text file $input with model $model to $type graphs, output in $output"

# create filename for amconll file
output=$output"/"
amconll=$output$type".amconll"

# run neural net + fixed-tree decoder to obtain AMConLL file. Pass the --give_up option if we want things to run faster.
if [ "$fast" = "false" ]; then
    python3 parse_raw_text.py $model $type $input $amconll --cuda-device $gpu
else
    python3 parse_raw_text.py $model $type $input $amconll --cuda-device $gpu --give_up 5
fi

# convert AMConLL file (consisting of AM depenendcy trees) to final output file (containing graphs in the representation-specific format)
echo "converting AMConLL to final output file .."
# TODO possibly clean up the if-then-else
if [ "$type" = "DM" ] || [ "$type" = "PAS" ]; then
    java -cp $jar de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus -c $amconll -o $output$type
else
    if [ "$type" = "PSD" ]; then
        java -cp $jar de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPCorpus -c $amconll -o $output$type
    else
        if [ "$type" = "EDS" ]; then
             java -cp $jar de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus -c $amconll -o "$output"$type
        fi
    fi
fi
