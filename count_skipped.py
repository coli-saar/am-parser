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
from typing import Dict, Any, List, Tuple
import logging
import json


from graph_dependency_parser.components.dataset_readers.amconll_tools import parse_amconll

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',level=logging.INFO) #turn on logging.

import argparse

parser = argparse.ArgumentParser(description="Count how many sentences don't have an AM dependency tree.")

parser.add_argument('input_file', type=str, help='amconll file')


args = parser.parse_args()

total = 0
skipped = 0
time_dict : Dict[str, List[float]] = dict()
with open(args.input_file) as f:
    for am_sentence in parse_amconll(f, validate=False):
        total += 1
        roots = [i for i in range(len(am_sentence)) if am_sentence.words[i].label == "ROOT" and am_sentence.words[i].head == 0]
        if len(roots) == 0:
            skipped += 1
        elif len(roots) == 1:
            root = am_sentence.words[roots[0]]
            if root.typ == "_":
                print("Annotation found but is a tree of \\bot")
                skipped +=1
            else:
                for k, v in am_sentence.attributes.items():
                    if "time" in k:
                        try:
                            f = float(v)
                            if k not in time_dict:
                                time_dict[k] = []
                            time_dict[k].append(f)
                        except ValueError:
                            pass
        else:
            print("Multiple roots?")

total_time = 0
print("="*80)
for k, v in time_dict.items():
    t = sum(v)
    total_time += t
    print(k, t)
print("="*80)
print("Time without skipped", total_time)
print("===")
print("Skipped", skipped)
print("Total", total)
print("Percentage skipped", round(skipped/total * 100,2),"%")