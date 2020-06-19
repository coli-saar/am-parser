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

mytmpdir=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`

python prepare_single_visualize.py "$1" $mytmpdir
b=`basename $1`
java -jar MaltEval.jar -g "$mytmpdir/$b.conllu" -v 1

rm -rf "$mytmpdir"
