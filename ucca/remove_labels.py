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
            
