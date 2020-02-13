#!/bin/bash

# TODO make this file safe for spaces in filenames

# default file paths
defaultmodel="downloaded_models/raw_text_model.tar.gz"
mkdir -p downloaded_models
jar="am-tools.jar"

# Documenting parameters:
usage="Takes . \n\n

Required arguments: \n
\n\t     -i  input file: text file with one sentence per line. By default, assumes the file is already tokenized with spaces as delimiters.
\n\t     -o  output folder: where the results will be stored.
\n\t     -T  desired type of output formalism. Possible options: DM, PAS, PSD, EDS and AMR-2017.

\noptions:

\n\t   -m  archived model file in .tar.gz format. If not given, the default model path $defaultmodel is used. If that
file does not exist, it will be downloaded automatically (NOT YET IMPLEMENTED)
\n\t	 -f  faster, less accurate (flag; default false)
\n\t	 -t  tokenize with spacy (flag; default false)
\n\t   -g  which gpu to use (its ID, i.e. 0, 1 or 2 etc). Default is -1, using CPU instead"

#defaults:
fast=false
gpu="-1"
tokenize=false
# Gathering parameters:
while getopts "m:i:o:T:g:fth" opt; do
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
	t) tokenize=true
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
        wget -O "$defaultmodel" http://www.coli.uni-saarland.de/projects/amparser/raw_text_model.tar.gz
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
    printf "\n No output graphbank type given. Please use -T option.\n"
    exit 1
elif [ "$type" = "AMR" ]; then
    type="AMR-2017" # to make it compatible with how we referred to this in the python code
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
    if [ "$tokenize" = "false" ]; then
      python3 parse_raw_text.py $model $type $input $amconll --cuda-device $gpu
    else
      python3 parse_raw_text.py $model $type $input $amconll --cuda-device $gpu --tokenize
    fi
else
    if [ "$tokenize" = "false" ]; then
      python3 parse_raw_text.py $model $type $input $amconll --cuda-device $gpu --give_up 5
    else
      python3 parse_raw_text.py $model $type $input $amconll --cuda-device $gpu --give_up 5 --tokenize
    fi
fi

# convert AMConLL file (consisting of AM depenendcy trees) to final output file (containing graphs in the representation-specific format)
echo "converting AMConLL to final output file .."
# TODO possibly clean up the if-then-else
if [ "$type" = "DM" ] || [ "$type" = "PAS" ]; then
    java -cp $jar de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus -c $amconll -o $output$type
elif [ "$type" = "PSD" ]; then
    java -cp $jar de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPCorpus -c $amconll -o $output$type
elif [ "$type" = "EDS" ]; then
    java -cp $jar de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus -c $amconll -o "$output"$type
elif [ "$type" = "AMR-2017" ]; then
    bash scripts/eval_AMR_new.sh $amconll $output $jar
fi
