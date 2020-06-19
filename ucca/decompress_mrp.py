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
from eliminate_h_top import add_h
from get_edges_from_mrp import get_id2lex, get_mrp_edges
from process_c import decompress_c
from convert_irtg_to_mrp import get_edges, get_tops
from test_compression_vis import update_id_labels



def get_terminal_nodes(mrp_dict):
    nodes = []
    for node in mrp_dict['nodes']:
        if node['label'] != 'Non-Terminal':
            nodes.append(node)
    return nodes



mrp = sys.argv[1]
out = sys.argv[2]

with open(out, 'w+') as outfile:
    with open(mrp) as infile:
        for line in infile:
            mrp_post_processed = {}
            mrp_dict = json.loads(line)
            input = mrp_dict['input']
            id = mrp_dict['id']
            #print(id)
            framework = mrp_dict['framework']
            flavor = mrp_dict['flavor']
            time = mrp_dict['time']
            version = mrp_dict['version']
            node_ids = get_id2lex(mrp_dict)
            edges = get_mrp_edges(mrp_dict)
            decompressed = decompress_c(edges, node_ids)
            decompressed = add_h(decompressed)
            mrp_post_processed['id'] = id
            mrp_post_processed['framework'] = framework
            mrp_post_processed['flavor'] = flavor
            mrp_post_processed['time'] = time
            mrp_post_processed['version'] = version
            mrp_post_processed['tops'] = get_tops(decompressed)
            label_dict,node_ids = update_id_labels(decompressed, node_ids)
            mrp_nodes = get_terminal_nodes(mrp_dict)
            for node in node_ids.keys():
                if node_ids[node] == 'Non-Terminal':
                    mrp_nodes.append({'id':node})
            mrp_post_processed['nodes'] = mrp_nodes
            mrp_post_processed['edges'] = get_edges(decompressed)
            mrp_post_processed['input'] = input
            outfile.write(json.dumps(mrp_post_processed))
            outfile.write('\n')
