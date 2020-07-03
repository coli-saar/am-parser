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
import os.path
from typing import Dict
import argparse


import sys
sys.path.append("..") # Adds higher directory to python modules path.

import graph_dependency_parser.components.dataset_readers.amconll_tools as amconll_tools

optparser = argparse.ArgumentParser(add_help=True,
    description="reads two amconll files and produces two conllu files (the intersection) sorted by size that can be compared with MaltEval")
optparser.add_argument("file1", type=str)
optparser.add_argument("file2", type=str)
optparser.add_argument("direc",help="where to store conllu files", type=str)

opts = optparser.parse_args()

with open(opts.file1) as f1:
    with open(opts.file2) as f2:
        sents1 : Dict[str,amconll_tools.AMSentence] = { sent.attributes["id"] : sent for sent in amconll_tools.parse_amconll(f1,validate=False) }
        sents2 : Dict[str,amconll_tools.AMSentence] = { sent.attributes["id"] : sent for sent in amconll_tools.parse_amconll(f2,validate=False) }

ids = sorted(set(sents1.keys()) & set(sents2.keys()), key=lambda id: len(sents1[id]))
with open(os.path.join(opts.direc,"1_"+os.path.basename(opts.file1)),"w") as of1:
    for id in ids:
        if sents1[id].get_tokens(shadow_art_root=False) != sents2[id].get_tokens(shadow_art_root=False) :
            print("Skipping",id,"because MaltEval would complain that text is different")
            continue
        for i,e in enumerate(sents1[id]):
            of1.write(str(i+1)+"\t"+"\t".join([e.token, e.lexlabel, e.fragment, e.typ,"_",str(e.head),e.label,"_","_"]))
            of1.write("\n")
        of1.write("\n")

with open(os.path.join(opts.direc, "2_"+os.path.basename(opts.file2)), "w") as of2:
    for id in ids:
        if sents1[id].get_tokens(shadow_art_root=False) != sents2[id].get_tokens(shadow_art_root=False) :
            continue
        for i,e in enumerate(sents2[id]):
            of2.write(str(i+1)+"\t"+"\t".join([e.token, e.lexlabel, e.fragment, e.typ,"_",str(e.head),e.label,"_","_"]))
            of2.write("\n")
        of2.write("\n")


