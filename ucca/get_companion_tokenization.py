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
import json
import re
import os

companion_data_dir = sys.argv[1]
outdir = sys.argv[2]
hyphen_re = re.compile(r'[–—−]')
for filename in os.listdir(companion_data_dir):
    mrp_companion = {}
    with open(companion_data_dir+filename) as infile:
        for line in infile:
            if line.startswith('#'):
                id = line[1:].strip()
                mrp_companion[id] = {'tokenization':[], 'spans':{}}
            elif not line.startswith('\n'):
                line = line.split()
                token = line[1]
                token = re.sub(r'[–—−]', '-', token).lower()
                if "’" in token:
                    token = token.replace("’", "'")
                #if "”" in token:
                #    token = token.replace("”", '"')
                #if "“" in token:
                #    token = token.replace("“", '"')
                if "…" in token:
                    token = token.replace("…", "...")
                #if '-' in token:
                #    token = token.replace('-', '—')
                mrp_companion[id]['tokenization'].append(token)
                span = line[-1][11:]
                index = line[0]
                mrp_companion[id]['spans'][span] = int(index) -1
    json.dump(mrp_companion, open(outdir+ filename[:-7]+'.json', 'w', encoding='utf8'), ensure_ascii=False)
