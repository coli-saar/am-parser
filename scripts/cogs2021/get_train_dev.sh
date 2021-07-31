#!/usr/bin bash
##
## Copyright (c) 2021 Saarland University.
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
# getting train and dev zip files for training the am-parser and also dev amconll file for evaluation
# todo test command line parsing: especially for erroneous input
# bash ./scripts/cogs2021/get_train_dev.sh -t ~/HiwiAK/cogs2021/small/train50.tsv -d ~/HiwiAK/cogs2021/small/dev10.tsv -o ~/HiwiAK/cogs2021/toy_model_run/training_input/
# bash ./scripts/cogs2021/get_train_dev.sh -t ~/HiwiAK/cogs2021/small/train50.tsv -d ~/HiwiAK/cogs2021/small/dev10.tsv -o ~/HiwiAK/cogs2021/toy_model_run/training_input/ -s 3 -p dp_dev
# and for preposition reification add the option  -r  at the end


jar="am-tools.jar"
# i.e. assumed to be in current directory, maybe cd to am-parser directory first

# Documenting parameters:
usage="Takes . \n\n

Required arguments: \n
\n\t     -t  train file: Path to the train file in the TSV format of COGS.
\n\t     -d  dev file: Path to the dev file in the TSV format of COGS.
\n\t     -o  output folder: where the results will be stored.

\noptions:

\n\t   -s  number of sources (default: 3).
\n\t   -p  dev eval prefix (default: dp_dev).
\n\t   -r flag to enable preposition reification (default: false)
"

#defaults:
prefix="dp_dev"
sources=3
reifypreps=false

# Gathering parameters:
# note: although -t,-d,-o are basically mandatory, we don't use positional
# arguments: hopefully easier to use and not confuse positions.
while getopts "t:d:o:s:p:rh" opt; do
    case $opt in
  h) echo -e $usage
     exit
     ;;
  t) train="$OPTARG"
     ;;
  d) dev="$OPTARG"
     ;;
  o) output="$OPTARG"
     ;;
  s) sources="$OPTARG"
     ;;
  p) prefix="$OPTARG"
     ;;
  r) reifypreps=true
     ;;
  \?) echo "Invalid option -$OPTARG" >&2
      ;;
    esac
done

# input validation:
if [ -f "$jar" ]; then
    echo "jar file found at $jar."
else
    echo "jar file not found at $jar."
    exit 1
fi

if [ -f "$train" ]; then
    echo "train file found at $train"
else
    echo "train file not found at $train. Please check the -t parameter"
    exit 1
fi

if [ -f "$dev" ]; then
    echo "dev file found at $dev"
else
    echo "dev file not found at $dev. Please check the -d parameter"
    exit 1
fi

if [ "$output" = "" ]; then
    printf "\nERROR: No output folder path given. Please use -o option.\n"
    exit 1
fi

echo "INFO: Number of sources: $sources"
if [ $sources -lt 1 ]; then
    prinf "\nERROR: source smaller 1 not allowed.\n"
    exit 1
fi

echo "INFO: Output prefix: $prefix"
if [ "$prefix" = "" ]; then
    echo "Empty prefix not allowed."
    exit 1
fi

echo "INFO: Preposition reification?: $reifypreps"

## Finally the interesting part:

printf "\n--> Automata for train ($train) and dev ($dev) : will create zip files in $output: ...\n"
if [ "$reifypreps" = "false" ]; then
    java -cp $jar de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS --trainingCorpus $train --devCorpus $dev --outPath $output --nrSources $sources --algorithm automata
else
    java -cp $jar de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS --trainingCorpus $train --devCorpus $dev --outPath $output --nrSources $sources --algorithm automata --reifyprep
fi

printf "\n--> Prepare dev data for evaluation ($prefix amconll file in $output will be created from $dev) ...\n"
java -cp $jar de.saar.coli.amrtagging.formalisms.cogs.tools.PrepareDevData --corpus $dev --outPath $output --prefix $prefix

printf "\nDone!\n"
