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
import json
import sys
from get_edges_from_mrp import get_id2lex, get_mrp_edges


corpus = sys.argv[1]

with open(corpus) as infile:
    for line in infile:
        mrp = json.loads(line)
        labels = get_id2lex(mrp)
        edges = get_mrp_edges(mrp)
        nodes = set()
        node_mentions_in_edges = set()
        for node_id in labels.keys():
            nodes.add(node_id)
        for (u, v) in edges.keys():
            node_mentions_in_edges.add(u)
            node_mentions_in_edges.add(v)
