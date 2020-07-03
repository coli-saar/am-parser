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
from get_edges_from_mrp import get_mrp_edges,get_id2lex, test
from process_c import compress_c_edge, decompress_c
from edge_to_irtg import edge2irtg
import json
import sys



infile = sys.argv[1]
def update_id_labels(edge_dict, label_dict):
    for (u,v) in list(edge_dict.keys()):
        if type(u) == str:
            label_dict[int(u.split('<')[0])] = '<root>'
            edge_dict[(int(u.split('<')[0]), v)] = edge_dict[(u,v)]
            del edge_dict[(u,v)]
        elif u - 1111 >= 0:
            if int(str(u)[:-4]) in label_dict.keys():
                label_dict[u] = label_dict[int(str(u)[:-4])]
            else: label_dict[u] = 'Non-Terminal'
    nodes_in_edge_dict = list(set([node for edge in edge_dict.keys() for node in edge]))
    label_dict_nodes = list(label_dict.keys())
    for edge in edge_dict.keys():
        for node in edge:
            if node not in label_dict.keys() and type(node) != str:
                label_dict[node] = 'Non-Terminal'
    return label_dict, edge_dict



'''non_deducible = ["id", "flavour", "framework", "version", "time"]
with open(infile, 'r') as f:
    for line in f:
        mrp_dict = json.loads(line)
        print(mrp_dict["id"])
        edges = get_mrp_edges(mrp_dict)
        labels = get_id2lex(mrp_dict)
        irtg_format_original = edge2irtg(edges, labels)
        print(irtg_format_original)

        compressed_edges = compress_c_edge(edges)
        compressed_labels = update_id_labels(compressed_edges, labels)
        irtg_format_compressed = edge2irtg(compressed_edges, labels)

        decompressed = decompress_c(compressed_edges, compressed_labels)
        decompressed_labels = update_id_labels(decompressed, compressed_labels)
        print(decompressed_labels)
        irtg_format_decompressed = edge2irtg(decompressed, decompressed_labels)

        print(irtg_format_compressed)
        print(irtg_format_decompressed)
        print("________________________________________-")'''
