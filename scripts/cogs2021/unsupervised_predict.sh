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

type="COGS"

jar="am-tools.jar"

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
  p) projective=true
     ;;
  \?) echo "Invalid option -$OPTARG" >&2
      ;;
    esac
done

if [ "$gpu" = "-1" ]; then
    echo "Warning: using CPU, this may be slow. Use -g to specify a GPU ID"
fi

if [ -f "$model" ]; then
    echo "model file found at $model"
else
    echo "model not found at $model. Please check the -m parameter"
    exit 1
fi

if [ -f "$jar" ]; then
    printf "jar file found at $jar\n"
else
    echo "jar file not found at $jar\n"
    exit 1
fi

if [ "$input" = "" ]; then
    printf "\n No input file given. Please use -i option.\n"
    exit 1
fi

if [ "$output" = "" ]; then
    printf "\n No output folder path given. Please use -o option.\n"
    exit 1
fi

# Finished gathering parameters. We are now guaranteed to have the necessary arguments stored in the right place.
echo "Parsing input file $input with model $model to $type graphs, output in $output"

# create filename for amconll file
output=$output"/"
prefix=$type"_gold"
amconll_input=$output$prefix".amconll" # used as input for neural model, but we must create it first
amconll_prediction=$output$type"_pred.amconll" # where the neural model writes its prediction

# convert input file to AMConLL format
echo "--> Convert input file to AMConLL format ..."
java -cp $jar de.saar.coli.amrtagging.formalisms.cogs.tools.PrepareDevData -c $input -o $output -p $prefix

if [ "$projective" = "false" ]; then
  # run neural net + fixed-tree decoder to obtain AMConLL file. Pass the --give_up option if we want things to run faster.
  # (pw: opened a github issue that using one thread seems to be faster and idk why, therefore using --thread 1 here)
  echo "--> Predicting with fixed-tree decocer (fast? $fast)..."
  if [ "$fast" = "false" ]; then
      python3 parse_file.py $model $type $amconll_input $amconll_prediction --cuda-device $gpu --threads 1
  else
      python3 parse_file.py $model $type $amconll_input $amconll_prediction --cuda-device $gpu --threads 1 --give_up 15
  fi
else
  echo "--> Predicting with projective A* decoder..."
  # todo maybe remove old logs/result files?
  echo " -> computing scores..."
  scoreszip=$output"scores.zip"
  # see  https://github.com/coli-saar/am-parser/wiki/Computing-scores
  # python dump_scores.py models/a_model <formalism> <input data.amconll> <output file.zip> --cuda-device 0
  # todo parameters to check:
  # --supertag-limit 15 # How many labels per edge to include in the scores file
  # --edge-label-limit 30 # How many labels per edge to include in the scores file
  python3 dump_scores.py $model $type $amconll_input $scoreszip --cuda-device $gpu
  echo " -> A* parsing..."
  # see https://github.com/coli-saar/am-parser/wiki/A*-Parser
  # java -cp <am-tools.jar> de.saar.coli.amtools.astar.Astar -s <scores.zip> -o <outdir>
  # --outside-estimator static  (static, trivial, supertagonly, root_aware, ignore_aware)
  # --threads <N>
  # --statistics <statistics.csv>   #runtime stats
  # NOTE: in cogs branch am-tools.jar version: Astar is still in de.saar.coli.irtg.experimental.astar.Astar
  # java -cp master-am-tools.jar de.saar.coli.amtools.astar.Astar --outside-estimator ignore_aware -s $scoreszip -o $output
  java -cp $jar de.saar.coli.irtg.experimental.astar.Astar  --outside-estimator ignore_aware -s $scoreszip -o $output
  # will produce a log_*.txt and a results_*.amconll
  # rename results file to amconll prediction filename (COGS_pred.amconll)
  resultsfile=$output"results_*.amconll"
  mv $resultsfile $amconll_prediction
  # todo speed up with typechache.dat and serialized score reader needed?
fi

echo "--> Done with predicting at time:"
date
# convert AMConLL file (consisting of AM dependency trees) to final output file (containing graphs in the representation-specific format)
# and evaluate
echo "--> converting AMConLL to final output file .."
java -cp $jar de.saar.coli.amrtagging.formalisms.cogs.tools.ToCOGSCorpus -c "$amconll_prediction" -o "$output$type""_pred.tsv" --gold "$input" --verbose

echo "--> DONE!"
# THE END
