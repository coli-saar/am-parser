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
# see also : predict.sh in main folder
# heavily copied todo rather modify predict.sh?
# COGS specific for now
# bash ./scripts/cogs2021/unsupervised_predict.sh -i ../cogs2021/small/test5.tsv -o ../cogs2021/toy_model_run/prediction_output -m ../cogs2021/toy_model_run/training_output/model.tar.gz -g 0 -f &> ../cogs2021/toy_model_run/prediction.log
# bash ./scripts/cogs2021/unsupervised_predict.sh -i ../cogs2021/small/test5.tsv -o ../cogs2021/decoding/test -m ../cogs2021/temp/model.tar.gz -g 0 -p &> ../cogs2021/decoding/predict-sh.log
# todo: for Astar need additional master am-tools jar: check existence and correct call


set -e # tells Bash to exit the script immediately if any command returns a non-zero status.
# set -x  (for debugging) Print commands and their arguments as they are executed.  (with set +x disabled), or globally as bash -x scriptname

type="COGS"  # hard-coded commands for cogs format (prepare dev, ...)

jar="am-tools.jar"

# optionally use additional jar from main branch (bug-free astar implementation, e.g. punctuation fix)
# use_second_jar=false  # use (buggy) code version of the cogs branch. Not recommended!
use_second_jar=true
astar_jar="master-am-tools.jar"

## __Command line parameters: definition and parsing__
# Documenting parameters:
usage="Takes . \n\n

Required arguments: \n
\n\t     -i  input file:  Graph corpus in the original format. For COGS this is .tsv format
\n\t     -o  output folder: where the results will be stored.

\noptions:

\n\t   -m  archived model file in .tar.gz format.
\n\t   -f  faster, less accurate evaluation (flag; default false) - only useful for fixed-tree decoder
\n\t   -p  use projective A* decoder instead of fixed-tree decoder (flag; default false)
\n\t   -g  which gpu to use (its ID, i.e. 0, 1 or 2 etc). Default is -1, using CPU instead"

#defaults:
fast=false
projective=false
gpu="-1"
# Gathering parameters:
while getopts "m:i:o:T:g:fph" opt; do
    case $opt in
  h) echo -e "$usage"
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
  p) projective=true
     ;;
  \?) echo "Invalid option -$OPTARG" >&2
      ;;
    esac
done

## __Helper functions__
function echoinfo () { echo "INFO: $@"; }
function echoerr () { 1>&2 printf "ERROR: %s\n" "$@"; }
function printAndEval() {
  # prints command and then evaluates it, use like  printAndEval "python3 --version"
  echoinfo "Now running:  $1"
  eval "$1";
}
function exitOnFileNotFound () {  # FILEPATH FILEDESC ADDITIONALHELP=''
    if [ -f "$1" ]; then  # check that exists and is regular file
        echoinfo "$2 found at $1"
    else
        echoerr "$2 not found at $1 . $3"
        exit 1
    fi;
}

## __Start actual script__
starttime=$(date +"%F %T %Z")
echoinfo "$0 started at $starttime"  # get_train_dev.sh started at 2020-01-31 14:01:52 CEST
echoinfo "$0 started with parameters $@"  # get_train_dev.sh started with parameters -t bla - d bla2 ...

## __Input validation__
exitOnFileNotFound $jar "am-tools.jar file"
if [ "$projective" == true ] && [ "$use_second_jar" == true ]; then
    echoinfo "For envoking Astar code use ADDITIONAL jar file: $astar_jar"
    exitOnFileNotFound $astar_jar "astar am-tools.jar file"
elif [ "$projective" == true ] && [ "$use_second_jar" != true ]; then
    echoinfo "For envoking Astar code DOES NOT use additional jar file (warning: buggy code?)"
fi
exitOnFileNotFound "$model" "model file" "Please check the -m parameter"
exitOnFileNotFound "$input" "input file" "Please check the -i parameter"

if [ -d "$output" ]; then
    echoinfo "Output folder found at $output"
else
    echoerr "Output folder not found: $output . Please check the -o parameter"
    exit 1
fi

if [ "$gpu" == "-1" ]; then
    echo "WARNING: using CPU, this may be slow. Use -g to specify a GPU ID"
fi

# Finished gathering parameters. We are now guaranteed to have the necessary arguments stored in the right place.
echoinfo "Parsing input file $input with model $model to $type graphs, output in $output"

## __Prepare filenames in variables__
# create filename for amconll file
output=$output"/"
prefix=$type"_gold"  # e.g. COGS_gold
amconll_input=$output$prefix".amconll" # /PATH/TO/COGS_gold.amconll : used as input for neural model, but we must create it first
amconll_prediction=$output$type"_pred.amconll" # /PATH/TO/COGS_pred.amconll : where the neural model writes its prediction
tsv_prediction=$output$type"_pred.tsv"  # /PATH/TO/COGS_pred.tsv : converted predictions from amconll to tsv format

## __convert input file to AMConLL format ('empty' amconll)__
echoinfo "--> Convert input file to AMConLL format ..."
currentCMD="java -cp $jar de.saar.coli.amrtagging.formalisms.cogs.tools.PrepareDevData -c $input -o $output -p $prefix"
printAndEval "$currentCMD"

## __use projective Astar or fixed-tree decoder to predict__
echoinfo "Projective? $projective"
if [ "$projective" == false ] ; then
  # ----> fixed-tree
  # run neural net + fixed-tree decoder to obtain AMConLL file. Pass the --give_up option if we want things to run faster.
  # (pw: opened a github issue that using one thread seems to be faster and idk why, therefore using --thread 1 here)
  echoinfo "--> Predicting with fixed-tree decocer (fast? $fast)..."
  currentCMD="python3 parse_file.py $model $type $amconll_input $amconll_prediction --cuda-device $gpu --threads 1"
  if [ "$fast" == true ]; then
      currentCMD+=" --give_up 15"
  fi
  printAndEval "$currentCMD"
else
  # ----> projective Astar
  echoinfo "--> Predicting with projective A* decoder..."
  # todo maybe remove old logs/result files?
  echoinfo " -> computing scores..."
  scoreszip=$output"scores.zip"
  # see  https://github.com/coli-saar/am-parser/wiki/Computing-scores
  # python dump_scores.py models/a_model <formalism> <input data.amconll> <output file.zip> --cuda-device 0
  # todo parameters to check:
  # --supertag-limit 15 # How many supertags per token to include in the scores file
  # --edge-label-limit 30 # How many labels per edge to include in the scores file
  currentCMD="python3 dump_scores.py $model $type $amconll_input $scoreszip --cuda-device $gpu"
  printAndEval "$currentCMD"
  echoinfo " -> A* parsing..."
  # see https://github.com/coli-saar/am-parser/wiki/A*-Parser
  # java -cp <am-tools.jar> de.saar.coli.amtools.astar.Astar -s <scores.zip> -o <outdir>
  # --outside-estimator static  (static, trivial, supertagonly, root_aware, ignore_aware)
  # --threads <N>
  # --statistics <statistics.csv>   #runtime stats
  # java -cp $jar de.saar.coli.irtg.experimental.astar.Astar --outside-estimator ignore_aware -s $scoreszip -o $output
  # java -cp master-am-tools.jar de.saar.coli.amtools.astar.Astar --outside-estimator ignore_aware -s $scoreszip -o $output
  if [ "$use_second_jar" == true ]; then
      currentCMD="java -cp $astar_jar de.saar.coli.amtools.astar.Astar"
  else
      currentCMD="java -cp $jar de.saar.coli.irtg.experimental.astar.Astar"
  fi
  currentCMD+=" --outside-estimator ignore_aware -s $scoreszip -o $output"
  printAndEval "$currentCMD"
  # will produce a log_*.txt and a results_*.amconll
  # rename results file to amconll prediction filename (COGS_pred.amconll)
  resultsfile=$output"results_*.amconll"
  mv $resultsfile $amconll_prediction
  # todo speed up with typechache.dat and serialized score reader needed?
fi
echoinfo "--> Done with predicting at time $(date)"

# __AMConLL to final format and evaluation__
# convert AMConLL file (consisting of predicted AM dependency trees) to the
# final output file (containing graphs in the representation-specific format)
# and evaluate using the gold tsv file
# todo use verbose option? yes or no?
echoinfo "--> converting AMConLL to final output file .."
currentCMD="java -cp $jar de.saar.coli.amrtagging.formalisms.cogs.tools.ToCOGSCorpus -c $amconll_prediction -o $tsv_prediction --gold $input --verbose"
printAndEval "$currentCMD"

printf "\n--> Done!  (End time: %s)\n" "$(date  +'%F %T %Z')"
# THE END
