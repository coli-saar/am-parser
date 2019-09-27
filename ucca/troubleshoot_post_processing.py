import sys
import json

from get_edges_from_mrp import get_id2lex, get_mrp_edges
from edge_to_irtg import edge2irtg
from process_c import *
from move_edges import lower_edge, raise_edge
from test_head_percolation import update_id_labels
from utils import number_edges
from a_star_mrp import get_roots

mrp_in = sys.argv[1]


with open(mrp_in) as infile:
    for line in infile:
        mrp = json.loads(line)
        id = mrp['id']
        #print(id)
        #print('original')
        labels = get_id2lex(mrp)
        edges = get_mrp_edges(mrp, get_remote = False)
        irtg_original = edge2irtg(edges, labels)
        #print(irtg_original)
        lowered = lower_edge(edges)
        decompressed = decompress_c(edges, labels)
        labels = update_id_labels(lowered, labels)
        irtg_decompressed = edge2irtg(decompressed, labels)
        #print('Lowered')
        #print(irtg_decompressed)
        if len(get_roots(decompressed)) == 0:
            print(get_roots(decompressed))
            print(id)
            print(irtg_decompressed)
        #print('_'*40)
