
import argparse
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from typing import Dict
import random

import os
import shutil

import graph_dependency_parser.components.dataset_readers.amconll_tools as amconll_tools

optparser = argparse.ArgumentParser(add_help=True, 
    description="Given an amconll file and a file with ids (one per line), extract the am dependency trees")
optparser.add_argument("amconll", help="input amconll corpus", type=str)
optparser.add_argument("ids", help="ids of sentences to include", type=str)
optparser.add_argument("output", help="output corpus", type=str)

args = optparser.parse_args()


with open(args.amconll) as fil:
    sents = { sent.attributes["id"] : sent for sent in amconll_tools.parse_amconll(fil,validate=False) }
    
subset = []
with open(args.ids) as f:
    for line in f:
        id = line.rstrip()
        if id not in sents:
            print("Couldn't find sentence for id",id)
            sys.exit()
        subset.append(sents[id])
        
with open(args.output,"w") as f:
    for sent in subset:
        f.write(str(sent))
        f.write("\n\n")
    
