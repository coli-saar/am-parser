import sys
import json
import collections

from edge_to_irtg import edge2irtg
from get_edges_from_mrp import get_id2lex, get_mrp_edges
from convert_irtg_to_mrp import get_edges, get_input, get_mrp_edges, get_nodes, get_tops, irtg2mrp
from eliminate_h_top import eliminate_h
from a_star_mrp import *
from process_c import *
from head_percolation_alignment import percolate


priority = 'L LR LA H P S A N C D T E R F G Q U'.split()
#priority_dict = {label:index for (index, label) in enumerate(labels)}
non_deducible = ["id", "flavour", "framework", "version", "time"]
mrp_data_path = sys.argv[1]
#companion_data = json.load(open(sys.argv[2], 'r', encoding = 'utf8'))
#outdir = sys.argv[3]

def update_id_labels(edge_dict, label_dict):
    for (u,v) in edge_dict.keys():
        if type(u) == str:
            if '<root>' in u:
                label_dict[u] = 'Non-Terminal'
        elif type(u) == str:
            label_dict[u] = u
        elif u - 1111 >= 0:
            if int(str(u)[:-4]) in label_dict.keys():
                label_dict[u] = label_dict[int(str(u)[:-4])]
            else: label_dict[u] = 'Non-Terminal'
    nodes_in_edge_dict = list(set([node for edge in edge_dict.keys() for node in edge]))
    label_dict_nodes = list(label_dict.keys())
    for edge in edge_dict.keys():
        for node in edge:
            if node not in label_dict.keys():
                label_dict[node] = 'Non-Terminal'
    return label_dict


with open(mrp_data_path,encoding='utf8', errors='ignore') as infile:
    counter = 0
    for line in infile:
        #print(line)
        mrp_dict = json.loads(line)
        id = mrp_dict["id"]
        print(id)
        edges = get_mrp_edges(mrp_dict)
        edges = eliminate_h(edges)
        labels = get_id2lex(mrp_dict)
        compressed_edges = compress_c_edge(edges)
        compressed_labels = update_id_labels(compressed_edges, labels)
        irtg_format_compressed = edge2irtg(compressed_edges, labels)
        print(irtg_format_compressed)
        #node_tokens = node_to_token_index(companion_data, mrp_dict, compressed_labels, id)
        #print(companion_data)
        #print(compressed_labels)
        #print(node_tokens)
        print(compressed_labels)
        alignments = percolate(compressed_edges, priority, compressed_labels)
        print(alignments)
        print('_________________________________________________-')
