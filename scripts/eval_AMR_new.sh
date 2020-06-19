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
amcoll=$1
dir=$2 # output directory
ALTO_PATH=$(readlink -f "$3")

#Call like this:
#bash scripts/eval_AMR_new.sh <input.amconll> <outdir> <path to am-tools.jar>


# if these files are missing, run scripts/setup_AMR.sh
lookup="downloaded_models/lookup/lookupdata17/"
wordnet_path="downloaded_models/wordnet3.0/dict/"

echo "Evaluating to graphs and relabelling with threshold 10"

java -Xmx2G -cp "$ALTO_PATH" de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus -c "$1" -o "$dir" --relabel --wn "$wordnet_path" --lookup "$lookup" --th 10

