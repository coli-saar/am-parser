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
        decompressed_c = decompress_c(edges)
        raised_d = raise_edge(decompressed_c, 'D', ['P', 'S'], mark =True)
        print(labels)
        for (u, v) in decompressed_c.keys():
            if u not in labels:
                labels[u] = 'Non-Terminal'
        print(labels)
        postprocessed_mrp = irtg2mrp(raised_d, labels)
        for key in extras.keys():
            postprocessed_mrp[key] = extras[key]
        with open(outfile, 'a') as out:
            out.write(json.dumps(postprocessed_mrp))
            out.write('\n')
