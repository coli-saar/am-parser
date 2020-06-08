"""
Takes several amconll files and computes the intersection of the ids.
"""

import argparse
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from typing import Dict
import random

import os
import shutil

import graph_dependency_parser.components.dataset_readers.amconll_tools as amconll_tools

optparser = argparse.ArgumentParser(add_help=True, 
    description="Takes random subsets of sizes 100, 1000, 10_000")
optparser.add_argument('--corpora',
                        nargs="+",
                        default=[],
                        help='Corpora to read')
optparser.add_argument("output_path", help="output path", type=str)

args = optparser.parse_args()


files = []
formalisms = []
for f in args.corpora:
    with open(f) as fil:
        sents : Dict[str,amconll_tools.AMSentence] = { sent.attributes["id"] : sent for sent in amconll_tools.parse_amconll(fil,validate=False) }
        files.append(sents)

intersection_of_ids = set(files[0].keys())
for sents in files:
    intersection_of_ids = intersection_of_ids & set(sents.keys())
    
with open(args.output_path, "w") as f:
    for id in files[0]: #Try to somewhat keep the order
        if id in intersection_of_ids:
            f.write(id)
            f.write("\n")

