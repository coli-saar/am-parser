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
output=$1

#run smatch:
# add newlines to goldAMR.txt
sed ':a;N;$!ba;s/\n/\n\n/g' "$output/goldAMR.txt" > "$output/goldAMR_newlines.txt"

# compute smatch score
echo "smatching in $output"

python2 external_eval_tools/smatch/smatch.py -f "$output/parserOut.txt" "$output/goldAMR_newlines.txt" --significant 4 --pr | tee $output/smatch.txt

