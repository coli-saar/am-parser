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
        print(id)
        labels = get_id2lex(mrp)
        edges = get_mrp_edges(mrp, get_remote = True)
        edges = number_edges(edges, 'A')
        print('original')
        irtg_original = edge2irtg(edges, labels)
        print(irtg_original)
        compressed = compress_c_edge(edges)
        labels = update_id_labels(compressed, labels)
        print('COMPRESSED')
        irtg_compressed = edge2irtg(edges, labels)
        print(irtg_compressed)
        print('RAISED U')
        raised_u = raise_edge(compressed, 'U', ['L', 'H', 'P', 'S', 'A','D'], label_dict=labels)
        labels = update_id_labels(raised_u, labels)
        irtg_raised_u = edge2irtg(raised_u, labels)
        print(irtg_raised_u)
        print('RAISED F')
        raised_f = raise_edge(raised_u, 'F', ['L', 'H', 'P', 'S', 'A','A1','A2', 'A3', 'A4', 'A5', 'D'], label_dict=labels)
        labels = update_id_labels(raised_f, labels)
        irtg_raised_f = edge2irtg(raised_f, labels)
        print(irtg_raised_f)
        print('Raised D')
        raised_d = raise_edge(raised_f, 'D', ['L', 'H', 'P', 'S', 'A''A1','A2', 'A3', 'A4', 'A5'], label_dict=labels)
        labels = update_id_labels(raised_d, labels)
        irtg_raised_d = edge2irtg(raised_d, labels)
        print(irtg_raised_d)
        lowered = lower_edge(raised_d)
        irtg_lowered = edge2irtg(lowered, labels)
        print('Lowered')
        print(irtg_lowered)
        roots = get_roots(lowered)
        print('________________________________-')
