#!/bin/bash

# Documenting parameters:
usage="Takes . \n\n

Required arguments: \n
\n\t     -m  archived model file in .tgz format
\n\t     -i  input file: text file with one sentence per line. Assumes the file is already tokenized with spaces as delimiters.
\n\t     -o  output folder: where the results will be stored.
\n\t     -T  desired type of output formalism. Possible options: DM, PAS, PSD, EDS (this raw text version does not support AMR).

\noptions:

\n\t	 -f  faster, less accurate evaluation (flag; default false)"

#defaults:
fast=false
# Gathering parameters:
while getopts "m:i:o:T:f" opt; do
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
	f) fast=true
	\?) echo "Invalid option -$OPTARG" >&2
	    ;;
    esac
done

if [ "$model" = "" ]; then
    printf "\n No model. Please use -m option.\n"
    exit 1
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
amconll=$output"/"$type".amconll"

# run neural net + fixed-tree decoder to obtain AMConLL file. Pass the --give_up option if we want things to run faster.
if [ "$fast" = "false" ]; then
    python3 parse_raw_text.py $model $type $input $amconll
else
    python3 parse_raw_text.py $model $type $input $amconll --give_up 5
fi


