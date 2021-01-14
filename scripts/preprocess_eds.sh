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

usage="Preprocess an DM 2015 Shared task corpus, in ACL 2019 style.\n\n
Arguments: \n

\n\t     -d  main directory where corpus lives
\n\t     -o  directory where output files will be put (default: 'output' within main directory)
"

while getopts "d:m:o:h" opt; do
    case $opt in
	h) echo -e $usage
	   exit
	   ;;
	d) maindir="$OPTARG"
	   ;;
	o) outputPath="$OPTARG"
	   ;;
       	\?) echo "Invalid option -$OPTARG" >&2
	    ;;
    esac
done

if [ "$maindir" = "" ]; then
    printf "No main directory given. Please use -d option.\n"
    exit 1
else
    printf "Processing files in main directory $maindir\n"
fi

if [ "$outputPath" = "" ]; then
    printf "No output directory given. Using default: 'output' folder inside main directory.\n"
    outputPath="$maindir/output"
fi
printf "Placing output in $outputPath\n"

#checking for am-tools
alto="am-tools.jar"

if [ -f "$alto" ]; then
    printf "jar file found at $alto\n"
else
    printf "jar file not found at $alto, downloading it!\n"
    wget -O "$alto" http://www.coli.uni-saarland.de/projects/amparser/am-tools.jar
fi

mkdir -p $outputPath

log=$outputPath/preprocessLog.txt
rm -f $log

#converting to am-tools
java -cp $alto de.saar.coli.amrtagging.formalisms.eds.tools.CreateCorpus -c $maindir/real_train.amr.txt -o /$outputPath/ -p real_train &>> $log

printf "process complete\n"
