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
import sys
import re

before_revert = sys.argv[1]
after_revert = sys.argv[2]


with open(after_revert, 'w+') as outfile:
    with open(before_revert) as infile:
        alto = ''
        for line in infile:
            alto += line
        alto = alto.split('\n\n')
        print(alto[2])
        for graph in alto:
            graph = graph.split('\n')
            print(graph)
            for line in graph:
                if line == graph[-1]:
                    outfile.write(line)
                    outfile.write('\n')
                else:
                    line = line.replace("'","’")
                    line = line.replace("RIGHT_QUOTATION", "”")
                    line = line.replace("LEFT_QUOTATION","“")
                    line = line.replace("QUOTATION_MARK", '"')
                    line = line.replace("...", "…")
                    line = line.replace("LEFT_SQUARE_BRACKET", "[")
                    line = line.replace("RIGHT_SQUARE_BRACKET", "]")
                    line = line.replace("c_acute", "c")
                    outfile.write(line)
                    outfile.write('\n')
            outfile.write('\n')
