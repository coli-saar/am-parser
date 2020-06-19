#!/usr/bin/python3
#
# Copyright (c) 2020 Saarland University.
#
# This file is part of AM Parser
# (see https://github.com/coli-saar/am-parser/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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

MTOOL_COMMAND = "mtool" #set path to mtool command, probably something like python3 /something/mtool/main.py


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
    if instance["edges"]["g"] > 16:
        return False
    interested_in = "edges"
    if fscore(instance[interested_in]) < 0.8 and (instance[interested_in]["s"] > 0 or instance[interested_in]["g"] > 0 ):
        return True
    # ~ for subtask in instance:
        # ~ if fscore(instance[subtask]) < 0.8 and instance[subtask]["s"] > 0:
            # ~ return True
    return False
    

def populate_with_ids(filename, ids):
    graphs = dict()
    with open(filename) as f:
        for line in f:
            graph = json.loads(line)
            if graph["id"] in ids:
                graphs[graph["id"]] = line
    return graphs


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
    result = json.loads(result)
    scores = result["scores"]
    for id in scores:
        interest = is_interesting(scores[id])
        if interest:
            interesting_ids.append(id)

file_1_graphs = populate_with_ids(file1,interesting_ids)
gold_graphs = populate_with_ids(gold_file,interesting_ids)

with TemporaryDirectory() as direc:
    for id in interesting_ids:
        print(id)
        if opts.viz:
            gold_graph = json.loads(gold_graphs[id])
            if "input" in gold_graph:
                print(gold_graph["input"])
            f1mrp = os.path.join(direc,"f1.mrp")
            with open(f1mrp,"w") as f:
                f.write(file_1_graphs[id])
            f2mrp = os.path.join(direc,"f2.mrp")
            with open(f2mrp,"w") as f:
                f.write(gold_graphs[id])
            os.system(MTOOL_COMMAND+ " --read mrp --normalize all --write dot "+ f1mrp + " "+  os.path.join(direc,"f1.dot"))
            os.system("dot -Tpng "+os.path.join(direc,"f1.dot")+" -o"+ os.path.join(direc,"f1.png"))

            os.system(MTOOL_COMMAND+ " --read mrp --normalize all --write dot "+ f2mrp + " "+  os.path.join(direc,"f2.dot"))
            os.system("dot -Tpng "+os.path.join(direc,"f2.dot")+" -o"+ os.path.join(direc,"f2.png"))
            viz_cmd = "montage "+os.path.join(direc,"f1.png")+" " +os.path.join(direc,"f2.png")+" -geometry +0+0 x:"
            with subprocess.Popen([viz_cmd], shell=True) as proc:
                pass
        
    
        
        
