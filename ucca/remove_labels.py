#!/usr/bin/python3
"""
Removes node labels from UCCA MRP graphs (must be post-processed!)

who to blame: matthias
"""


import argparse
import json

optparser = argparse.ArgumentParser(add_help=True, 
    description="Removes node labels from UCCA MRP graphs (must be post-processed!)")
optparser.add_argument("input_file", help="input file", type=str)
optparser.add_argument("output_file", help="where to store output", type=str)


opts = optparser.parse_args()

def remove_labels(mrpgraph):
    for i in range(len(mrpgraph["nodes"])):
        if "label" in mrpgraph["nodes"][i]:
            del mrpgraph["nodes"][i]["label"]

with open(opts.input_file) as inf:
    with open(opts.output_file,"w") as of:
        for line in inf:
            mrpgraph = json.loads(line)
            remove_labels(mrpgraph)
            of.write(json.dumps(mrpgraph,ensure_ascii=False))
            of.write("\n")
            
