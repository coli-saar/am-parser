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
import ast
import json

from get_edges_from_mrp import get_id2lex, get_mrp_edges
from move_edges import lower_edge, raise_edge
from edge_to_irtg import edge2irtg
from convert_irtg_to_mrp import get_edges, get_id2lex, get_input, get_mrp_edges, get_nodes, get_tops, irtg2mrp
from process_c import compress_c_edge, decompress_c

infile = sys.argv[1]
outfile = sys.argv[2]


non_deducible = ["id", "flavor", "framework", "version", "time"]
with open(infile, 'r') as f:
    for line in f:
        mrp_dict = json.loads(line)
        extras = {}
        for category in mrp_dict.keys():
            if category in non_deducible:
                extras[category] = mrp_dict[category]
        edges = get_mrp_edges(mrp_dict)
        labels = get_id2lex(mrp_dict)
        compressable, total, compressed_c = compress_c_edge(edges)
        lowered_d = lower_edge(compressed_c, 'D', ['P', 'S'], mark =True)
        preprocessed_mrp = irtg2mrp(lowered_d, labels)
        for key in extras.keys():
            preprocessed_mrp[key] = extras[key]
        with open(outfile, 'a') as out:
            out.write(json.dumps(preprocessed_mrp)+'\n')
