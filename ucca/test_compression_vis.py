from get_edges_from_mrp import get_mrp_edges,get_id2lex, test
from process_c import compress_c_edge, decompress_c
from edge_to_irtg import edge2irtg
import json
import sys



infile = sys.argv[1]
def update_id_labels(edge_dict, label_dict):
    for (u,v) in edge_dict.keys():
        if type(u) == str:
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
