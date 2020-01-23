import os.path
from typing import Dict
import argparse


import sys
sys.path.append("..") # Adds higher directory to python modules path.

import graph_dependency_parser.components.dataset_readers.amconll_tools as amconll_tools

optparser = argparse.ArgumentParser(add_help=True,
    description="reads an amconll file and produces a conllu file")
optparser.add_argument("file", type=str)
optparser.add_argument("direc",help="where to store conllu file", type=str)

opts = optparser.parse_args()

with open(opts.file) as f1:
    with open(os.path.join(opts.direc,os.path.basename(opts.file)+".conllu"),"w") as of1:
        for sent in amconll_tools.parse_amconll(f1,validate=False):
            for i,e in enumerate(sent):
                of1.write(str(i+1)+"\t"+"\t".join([e.token, e.lexlabel, e.fragment, e.typ,"_",str(e.head),e.label,"_","_"]))
                of1.write("\n")
            of1.write("\n")

