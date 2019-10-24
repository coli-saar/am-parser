#!/bin/bash

mytmpdir=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`

python prepare_visualize.py "$1" "$1" $mytmpdir
b=`basename $1`
java -jar MaltEval.jar -g "$mytmpdir/1_$b" -v 1

rm -rf "$mytmpdir"
