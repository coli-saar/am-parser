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
# bash ./scripts/cogs2021/get_train_dev.sh -t TRAINFILE -d DEVFILE -o OUTDIR [ -s NUMBEROFSOURCES -p DEVAMCONLLPREFIX -e TESTFILE -r -h ]
# bash ./scripts/cogs2021/get_train_dev.sh -t ~/HiwiAK/cogs2021/small/train50.tsv -d ~/HiwiAK/cogs2021/small/dev10.tsv -o ~/HiwiAK/cogs2021/toy_model_run/training_input/
# bash ./scripts/cogs2021/get_train_dev.sh -t ~/HiwiAK/cogs2021/small/train50.tsv -d ~/HiwiAK/cogs2021/small/dev10.tsv -o ~/HiwiAK/cogs2021/toy_model_run/training_input/ -s 3 -p dp_dev
# and for preposition reification add the option  -r  at the end
# and for generation of a test.amconll in the output folder add  -e ~/HiwiAK/cogs2021/small/test50.tsv
# you can redirect all output to a log file using  &> PATH/TO/get_train_dev.log

set -e # tells Bash to exit the script immediately if any command returns a non-zero status.
# set -x  (for debugging) Print commands and their arguments as they are executed.  (with set +x disabled), or globally as bash -x scriptname

jar="am-tools.jar"
# i.e. assumed to be in current directory, maybe cd to am-parser directory first

## __Command line parameters: definition and parsing__
# Documenting parameters:
usage="Takes . \n\n

Required arguments: \n
\n\t     -t  train file: Path to the train file in the TSV format of COGS.
\n\t     -d  dev file: Path to the dev file in the TSV format of COGS.
\n\t     -o  output folder: where the results will be stored.

\noptions:

\n\t   -s  number of sources (default: 3).
\n\t   -p  dev eval prefix (default: dp_dev).
\n\t   -e  test file: Path to the test file in the TSV format of COGS (default: no file, so no test.amconll produced)
\n\t   -r flag to enable preposition reification (default: false)
"

#defaults:
testfile=""
prefix="dp_dev"
sources=3
reifypreps=false

# Gathering parameters:
# note: although -t,-d,-o are basically mandatory, we don't use positional
# arguments: hopefully easier to use and not confuse positions.
while getopts "t:d:o:s:p:e:rh" opt; do
    case $opt in
  h) echo -e "$usage"
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
  e) testfile="$OPTARG"
     ;;
  \?) echo "Invalid option -$OPTARG" >&2
      ;;
    esac
done

## __Helper functions__
function echoinfo () { echo "INFO: $@"; }
function echoerr () { 1>&2 printf "ERROR: %s\n" "$@"; }
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
exitOnFileNotFound "$train" "train file" "Please check the -t parameter"
exitOnFileNotFound "$dev" "dev file" "Please check the -d parameter"

if [ -d "$output" ]; then
    echoinfo "Output folder found at $output"
else
    echoerr "Output folder not found: $output . Please check the -o parameter"
    exit 1
fi

echoinfo "Number of sources: $sources"
if [ $sources -lt 1 ]; then
    echoerr "source smaller 1 not allowed."
    exit 1
fi

echoinfo "Output prefix: $prefix"
if [ "$prefix" = "" ]; then
    echoerr "Empty prefix not allowed."
    exit 1
fi

if [ "$testfile" = "" ]; then
    echoinfo "Test file? No test.amconll will be prepared"
else
    exitOnFileNotFound "$testfile" "test file" "Please check the optional -e parameter"
fi

echoinfo "Preposition reification?: $reifypreps"

## __Finally the interesting part__

# get train.zip and dev.zip
echoinfo "Automata for train ($train) and dev ($dev) : will create zip files in $output: ..."
currentCMD="java -cp $jar de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS --trainingCorpus $train --devCorpus $dev --outPath $output --nrSources $sources --algorithm automata"
if [ "$reifypreps" == true ]; then
    currentCMD+=" --reifyprep"
fi
echoinfo "Now executing:  $currentCMD"
eval "$currentCMD"

# get dp_dev.amconll (or whatever prefix used)
echoinfo "Prepare dev data for evaluation ($prefix amconll file in $output will be created from $dev) ..."
currentCMD="java -cp $jar de.saar.coli.amrtagging.formalisms.cogs.tools.PrepareDevData --corpus $dev --outPath $output --prefix $prefix"
echo "INFO: Now executing:  $currentCMD"
eval "$currentCMD"

# optional: get test.amconll (if test file was provided as cmd arg)
if [ "$testfile" != "" ]; then
    echoinfo "Prepare test data for evaluation (test.amconll file in $output will be created from $testfile) ..."
    currentCMD="java -cp $jar de.saar.coli.amrtagging.formalisms.cogs.tools.PrepareDevData --corpus $testfile --outPath $output --prefix test"
    echoinfo "Now executing:  $currentCMD"
    eval "$currentCMD"
fi

printf "\nDone!  (End time: %s)\n" "$(date  +'%F %T %Z')"
