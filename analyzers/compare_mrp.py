#!/usr/bin/python3
"""
Compare two MRP files by scoring them and showing interesting ones.


Setup:

1.) Set MTOOL_COMMAND, probably you want to have something like "python3 mtool/main.py"
2.) Define is_interesting in way that suits you

Usage:

./compare_mrp.py example_compare_mrp/system_1.mrp example_compare_mrp/system_2.mrp --mtool "--n 2 --score mrp"

will call mtool to evaluate the first two graphs and it will print the ids of graphs you consider interesting

The option --mtool can be used to pass on arguments to mtool.

If you have dot and https://imagemagick.org/index.php installed, you can also give it the option --viz
then it will directly visualize the graphs.

who to blame: matthias
"""

import sys
import os
from tempfile import TemporaryDirectory
import subprocess
import json

import argparse

MTOOL_COMMAND = "mtool" #set path to mtool command, possibly starting with python3 ...


def fscore(triple):
    if triple["s"] > 0:
        precision = triple["c"] / triple["s"]
    else:
        precision = 0
    if triple["g"] > 0:
        recall = triple["c"] / triple["g"]
    else:
        recall = 0
    if precision+recall == 0:
        return 0.0
    return 2*precision*recall/(precision+recall)

def is_interesting(instance):
    """
    Define what to filter for.
    """
    # instance might look like this:
    # this comes from the mrp --trace output and might look different for different evaluation metrics (here you see --score mrp):
    # {"tops": {"g": 1, "s": 1, "c": 1}, "labels": {"g": 6, "s": 6, "c": 6}, "properties": {"g": 1, "s": 1, "c": 1}, "anchors": {"g": 0, "s": 0, "c": 0}, "edges": {"g": 6, "s": 6, "c": 5}
    if instance["labels"]["g"] > 15:
        return False
    if fscore(instance["edges"]) < 0.6 and instance["edges"]["s"] > 0:
        return True
    # ~ for subtask in instance:
        # ~ if fscore(instance[subtask]) < 0.6 and instance[subtask]["s"] > 0:
            # ~ return True
    return False
    




optparser = argparse.ArgumentParser(add_help=True, 
    description="compares two mrp files and spits out list of ids with discrepencies/or visualizes them directly")
optparser.add_argument("file1", help="system output", type=str)
optparser.add_argument("gold_file", help="gold file", type=str)
optparser.add_argument("--mtool", type=str,
                       help="arguments for mtool")
optparser.add_argument("--viz", action="store_true",
                       help="directly visualize")
opts = optparser.parse_args()

file1 = opts.file1
gold_file = opts.gold_file

interesting_ids = []

CMD = MTOOL_COMMAND+ " --read mrp --trace --gold "+ gold_file + " "+opts.mtool+" "+file1

with subprocess.Popen([CMD], shell=True, stdout=subprocess.PIPE) as proc:
    result = bytes.decode(proc.stdout.read())  # output of shell commmand as string
    print(result)
    result = json.loads(result)
    scores = result["scores"]
    for id in scores:
        interest = is_interesting(scores[id])
        if interest:
            interesting_ids.append(id)

for id in interesting_ids:
    print(id)
    if opts.viz:
        with TemporaryDirectory() as direc:
            os.system(MTOOL_COMMAND+ " --read mrp --write dot --id "+id +" "+ file1 + " "+  direc+"/f1.dot")
            os.system("dot -Tpng "+direc+"/f1.dot -o"+ direc+"/f1.png")

            os.system(MTOOL_COMMAND+ " --read mrp --write dot --id "+id +" "+ gold_file + " "+  direc+"/f2.dot")
            os.system("dot -Tpng "+direc+"/f2.dot -o"+ direc+"/f2.png")
            os.system("montage "+direc+"/f1.png"+" " +direc+"/f2.png -geometry +0+0 x:")
        
    
        
        
