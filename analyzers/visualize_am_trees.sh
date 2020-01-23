#!/bin/bash

mytmpdir=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`

python prepare_single_visualize.py "$1" $mytmpdir
b=`basename $1`
java -jar MaltEval.jar -g "$mytmpdir/$b.conllu" -v 1

rm -rf "$mytmpdir"
