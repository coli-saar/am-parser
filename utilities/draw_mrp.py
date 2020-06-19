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
import os
import sys
import subprocess
import tempfile

data = sys.stdin.readlines()
data = "\n".join(data)


tmpfile = tempfile.mktemp()
tmpdot = tempfile.mktemp("dot")
tmppdf = tempfile.mktemp("pdf")

with open(tmpfile, "w") as f:
    f.write(data)

os.system(f"python ../mtool/main.py --n 1 --ids --strings --read mrp --write dot {tmpfile} {tmpdot}")
os.system(f"dot -Tpdf {tmpdot} > {tmppdf}")
os.system(f"open {tmppdf}")