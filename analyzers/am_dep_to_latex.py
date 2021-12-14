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

import parsers.components.dataset_readers.amconll_tools as amconll_tools

optparser = argparse.ArgumentParser(add_help=True,
    description="reads an amconll file and produces the LaTeX and dot files for a specified AM dependency tree, by default the first one in the file.")
optparser.add_argument("file", type=str)
optparser.add_argument("direc",help="where to store the LaTeX file", type=str)
optparser.add_argument("--id",help="id of sentence to visualize", type=str, default=None)
optparser.add_argument("--i",help="1-based index of sentence to visualize, only used if no id given", type=int, default=1)



opts = optparser.parse_args()

found = False
with open(opts.file) as f:
    for i,sent in enumerate(amconll_tools.parse_amconll(f,validate=False),1):
        if opts.id == sent.attributes["id"] or (opts.id is None and i == opts.i):
            sent.to_tex_svg(opts.direc)
            found = True
            break

if not found:
    print("Sorry, couldn't find your sentence.")
