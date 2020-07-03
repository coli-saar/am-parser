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
from get_edges_from_mrp import get_id2lex, get_mrp_edges
from process_c import decompress_c
from convert_irtg_to_mrp import get_edges, get_tops
from test_compression_vis import update_id_labels
from move_edges import lower_edge
from eliminate_h_top import add_h

mrp = sys.argv[1]
out = sys.argv[2]

def get_terminal_nodes(mrp_dict):
    nodes = []
    for node in mrp_dict['nodes']:
        if node['label'] != 'Non-Terminal':
            nodes.append(node)
    return nodes

def strip_edge_info(edge_dict):
    stripped_dict = {}
    for edge in edge_dict.keys():
        label = edge_dict[edge].split('-')[0]
        label = edge_dict[edge].split('_')[0]
        stripped_dict[edge] = label
    return stripped_dict


with open(out, 'w+') as outfile:
    with open(mrp) as infile:
        for line in infile:
            mrp_post_processed = {}
            mrp_dict = json.loads(line)
            input = mrp_dict['input']
            id = mrp_dict['id']
            framework = mrp_dict['framework']
            flavor = mrp_dict['flavor']
            time = mrp_dict['time']
            version = mrp_dict['version']
            tops = mrp_dict['tops']
            if len(tops) > 0 :
                top = tops[0]
            else:
                top = None
            node_ids = get_id2lex(mrp_dict)
            edges = get_mrp_edges(mrp_dict)
            lowered = lower_edge(edges)
            decompressed = decompress_c(lowered, node_ids)
            decompressed = strip_edge_info(decompressed)
            revised_top, with_h = add_h(decompressed, node_ids, top)
            mrp_post_processed['id'] = id
            mrp_post_processed['framework'] = framework
            mrp_post_processed['flavor'] = flavor
            mrp_post_processed['time'] = time
            mrp_post_processed['version'] = version
            if revised_top is None:
                mrp_post_processed['tops'] = []
            else:
                mrp_post_processed['tops'] = [revised_top]
            node_ids = update_id_labels(decompressed, node_ids)[0]
            mrp_nodes = get_terminal_nodes(mrp_dict)
            for node in node_ids.keys():
                if node_ids[node] == 'Non-Terminal':
                    mrp_nodes.append({'id':node})
            mrp_post_processed['nodes'] = mrp_nodes
            mrp_post_processed['edges'] = get_edges(with_h)
            mrp_post_processed['input'] = input
            #print(mrp_post_processed)
            outfile.write(json.dumps(mrp_post_processed))
            outfile.write('\n')
