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
Takes an MRP file and spits writes the MRP graphs of the devset into a second corpus.

who to blame: matthias
"""
import argparse
import json
import random
import sys

optparser = argparse.ArgumentParser(add_help=True, 
    description="creates dev split for MRP file and spits out ids of devset sentences")
optparser.add_argument("file", help="input MRP corpus", type=str)
optparser.add_argument("file2", help="MRP dev corpus, will be created", type=str)
optparser.add_argument("file3", help="MRP train corpus, will be created", type=str)
optparser.add_argument("--fraction", type=float,default=3.0,
                       help="what PERCENTAGE of corpus should (roughly) go into dev data?")
opts = optparser.parse_args()

random.seed(11)

with open(opts.file) as f:
    with open(opts.file2,"w") as f2:
        with open(opts.file3,"w") as f3:
            for line in f:
                sentence = json.loads(line)
                sample = random.random() * 100
                if sample < opts.fraction:
                    f2.write(line)
                else:
                    f3.write(line)
