from get_edges_from_mrp import get_mrp_edges,get_id2lex, test
from process_c import compress_c_edge, decompress_c
from edge_to_irtg import edge2irtg
import ast
import sys



infile = sys.argv[1]


non_deducible = ["id", "flavour", "framework", "version", "time"]
with open(infile, 'r') as f:
    for line in f:
        mrp_dict = ast.literal_eval(line)
        edges = get_mrp_edges(mrp_dict)
        labels = get_id2lex(mrp_dict)
        ####################################
        #print("original")
        irtg_format_original = edge2irtg(edges, labels)
        #print(irtg_format_original)

        #print("compressed")
        total_cs, compressable_cs, compressed_edges = compress_c_edge(edges)
        irtg_format_compressed = edge2irtg(compressed_edges, labels)
        #print(irtg_format_compressed)

        print("decompressed")
        decompressed = decompress_c(compressed_edges)

        labels_decompressed = {}
        for label in labels.keys():
            labels_decompressed[label] = labels[label]
        for (u, v) in decompressed.keys():
            if u in labels_decompressed.keys():
                pass
            else:
                labels_decompressed[u] = u

        irtg_format_decompressed = edge2irtg(decompressed, labels_decompressed)
        print(irtg_format_original)
        print(irtg_format_compressed)
        print(irtg_format_decompressed)
        print("________________________________________-")
