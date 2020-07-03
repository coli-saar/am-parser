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
Takes n random subsets of sizes 100, 1000, 10_000 from the intersection of some corpora.
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
optparser.add_argument("n", help="number of subsets", type=int)
optparser.add_argument('--corpora',
                        nargs="+",
                        default=[],
                        help='Corpora to read, path to folder which contains train/ gold-dev/ and the like.')
optparser.add_argument("output_dir", help="output path", type=str)

args = optparser.parse_args()

if args.n < 1:
    print("Number of times subsets are taken must be at least 1")
    sys.exit()


files = []
formalisms = []
for f in args.corpora:
    normpath = os.path.normpath(f)
    formalisms.append(os.path.basename(normpath))
    with open(os.path.join(f,"train","train.amconll")) as fil:
        sents : Dict[str,amconll_tools.AMSentence] = { sent.attributes["id"] : sent for sent in amconll_tools.parse_amconll(fil,validate=False) }
        files.append(sents)

intersection_of_ids = set(files[0].keys())
for sents in files:
    intersection_of_ids = intersection_of_ids & set(sents.keys())
    
intersection_of_ids = sorted(intersection_of_ids)
random.seed(13)

for i in range(1,args.n+1):
    random.shuffle(intersection_of_ids)
    for formalism, sents, source_dir in zip(formalisms, files, args.corpora):
        for subset_size in [100, 1000, 10_000]:
            path = os.path.join(args.output_dir, str(subset_size) + "_" + str(i), formalism)
            #Copy everything from source directory
            shutil.copytree(source_dir, path)
            # Overwrite copy of training data with subset of the training data:
            with open(os.path.join(path, "train", "train.amconll"),"w") as f:
                subset = sorted(intersection_of_ids[:subset_size])
                for id in subset:
                    f.write(str(sents[id]))
                    f.write("\n\n")
                
    
    

