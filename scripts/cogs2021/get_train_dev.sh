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
# but there are caveats: https://mywiki.wooledge.org/BashFAQ/105

jar="am-tools.jar"
# i.e. assumed to be in current directory, maybe cd to am-parser directory first

## __Command line parameters: definition and parsing__
# defaults:
testfile=""
prefix="dp_dev"
sources=3
reifyprepositions=false
uselexlabelrepl=false

# Documenting parameters:
showUsage() {
  # `cat << EOF` This means that cat should stop reading when EOF is detected
  cat << EOF
Takes

Required arguments:
     -t  train file: Path to the train file in the TSV format of COGS.
     -d  dev file: Path to the dev file in the TSV format of COGS.
     -o  output folder: where the results will be stored.

options:

   -s  number of sources (default: $sources).
   -p  dev eval prefix (default: $prefix).
   -e  test file: Path to the test file in the TSV format of COGS (default: no file, so no test.amconll produced)
   -r flag to enable preposition reification (default: no reification)
   -l flag to enable copy mechanism for lexical labels (adds extra lex label with meaning 'copy word form')
EOF
# EOF is found above and hence cat command stops reading. This is equivalent to echo but much neater when printing out.
}

# Gathering parameters:
# note: although -t,-d,-o are basically mandatory, we don't use positional
# arguments: hopefully easier to use and not confuse positions.
while getopts "t:d:o:s:p:e:rlh" opt; do
    case $opt in
  h) showUsage
     exit 0
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
  r) reifyprepositions=true
     ;;
  l) uselexlabelrepl=true
     ;;
  e) testfile="$OPTARG"
     ;;
  \?) echo "Invalid option -$OPTARG" >&2
      ;;
    esac
done

## __Helper functions__
printinfo () { echo "INFO: $@"; }
printerror () { 1>&2 printf "ERROR: %s\n" "$@"; }
printAndCall () {
  set -x;  # enabled printing every following command
  "$@";
  { set +x; } 2>/dev/null;  # stops printing, also don't print  set +x
}
exitOnFileNotFound () {  # FILEPATH FILEDESC ADDITIONALHELP=''
    if [ -f "$1" ]; then  # check that exists and is regular file
        printinfo "$2 found at $1"
    else
        printerror "$2 not found at $1 . $3"
        exit 1
    fi;
}

## __Start actual script__
starttime=$(date +"%F %T %Z")
printinfo "$0 started at $starttime"  # get_train_dev.sh started at 2020-01-31 14:01:52 CEST
printinfo "$0 started with parameters $@"  # get_train_dev.sh started with parameters -t bla - d bla2 ...

## __Input validation__
exitOnFileNotFound "$jar" "am-tools.jar file"
exitOnFileNotFound "$train" "train file" "Please check the -t parameter"
exitOnFileNotFound "$dev" "dev file" "Please check the -d parameter"

if [ -d "$output" ]; then
    printinfo "Output folder found at $output"
else
    printerror "Output folder not found: $output . Please check the -o parameter"
    exit 1
fi

printinfo "Number of sources: $sources"
if [ $sources -lt 1 ]; then
    printerror "source smaller 1 not allowed."
    exit 1
fi

printinfo "Output prefix: $prefix"
if [ "$prefix" = "" ]; then
    printerror "Empty prefix not allowed."
    exit 1
fi

if [ "$testfile" = "" ]; then
    printinfo "Test file? No test.amconll will be prepared"
else
    exitOnFileNotFound "$testfile" "test file" "Please check the optional -e parameter"
fi

printinfo "Preposition reification?: $reifyprepositions"
printinfo "Use lex label replacement (copy mechanism)?: $uselexlabelrepl"

## __Finally the interesting part__

# get train.zip and dev.zip
printinfo "Automata for train ($train) and dev ($dev) : will create zip files in $output: ..."
doDecomposition() {
  if [ "$reifyprepositions" = true ]; then  # todo this is an ugly hack
    addreify="dummy"  # variable only defined if reification enabled
  fi
  if [ "$uselexlabelrepl" = true ]; then  # todo this is an ugly hack
    addlexcopy="dummy"  # variable only defined if lex label replacement enabled
  fi
  printAndCall java -cp "$jar" de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS \
      --trainingCorpus "$train" \
      --devCorpus "$dev" \
      --outPath "$output" \
      --nrSources "$sources" \
      --algorithm automata \
      ${addreify:+--reifyprep} \
      ${addlexcopy:+--useLexLabelReplacement}
}
doDecomposition

# get dp_dev.amconll (or whatever prefix used)
printinfo "Prepare dev data for evaluation ($prefix amconll file in $output will be created from $dev) ..."
prepareEmptyAmconll() {
  printAndCall java -cp "$jar" de.saar.coli.amrtagging.formalisms.cogs.tools.PrepareDevData \
      --corpus "$1" \
      --outPath "$output" \
      --prefix "$2"
}
prepareEmptyAmconll "$dev" "$prefix"

# optional: get test.amconll (if test file was provided as cmd arg)
if [ "$testfile" != "" ]; then
    printinfo "Prepare test data for evaluation (test.amconll file in $output will be created from $testfile) ..."
    prepareEmptyAmconll "$testfile" "test"
fi

printf "\nDone!  (End time: %s)\n" "$(date  +'%F %T %Z')"
