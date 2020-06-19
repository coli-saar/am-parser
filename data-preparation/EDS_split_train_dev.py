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
This script takes EDS graphs in penman notation (AMR notation) and splits into training and dev set in the same way as
https://github.com/semantic-dependency-parsing/toolkit
on the Penn Tree Bank
"""

filename = "train.amr.txt"

out_train = "real_train.amr.txt"
out_dev = "dev.amr.txt"

lines = []
with open(filename) as f:
    with open(out_train,"w") as ft:
        with open(out_dev,"w") as fd:
            
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("# ::id"):
                    if (line[len("# ::id "):])[1:3] == "20":
                        this_f = fd
                    else:
                        this_f = ft
                if line == "":
                    this_f.write("\n".join(lines))
                    this_f.write("\n\n")
                    lines = []
                else:
                    lines.append(line)

                
            
