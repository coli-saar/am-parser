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

mrp_everything = sys.argv[1]
dev_data = sys.argv[2]
out = sys.argv[3]


mrp_stored = {}
with open(mrp_everything) as mrp_in:
    for line in mrp_in:
        mrp = json.loads(line)
        id = mrp['id']
        mrp_stored[id] = mrp

#dev_stored = {}
dev_string = ''
with open(dev_data) as dev_in:
    for line in dev_in:
        if not line.startswith('#'):
            dev_string += line
dev_list = dev_string.split('\n\n')
dev_ids = [line.split('\n')[0] for line in dev_list]

with open(out, 'w') as outfile:
    for id in mrp_stored.keys():
        if id in dev_ids:
            outfile.write(json.dumps(mrp_stored[id]))
            outfile.write('\n')
