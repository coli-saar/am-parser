#!/bin/bash
output=$2 # directory with goldAMR.txt and parserOut.txt

#run smatch:
# add newlines to goldAMR.txt
sed ':a;N;$!ba;s/\n/\n\n/g' "$output/goldAMR.txt" > "$output/goldAMR_newlines.txt"

# compute smatch score
echo "smatching in $output"

python2 external_eval_tools/smatch/smatch.py -f "$output/parserOut.txt" "$output/goldAMR_newlines.txt" --significant 4 --pr | tee $output/smatch.txt

