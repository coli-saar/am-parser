import sys
import json
import collections
import os
import random
from tqdm import tqdm

from edge_to_irtg import edge2irtg
from get_edges_from_mrp import get_id2lex, get_mrp_edges
from convert_irtg_to_mrp import get_edges, get_mrp_edges, get_nodes, get_tops, irtg2mrp
from eliminate_h_top import eliminate_h
from a_star_mrp import *

mrp_in = sys.argv[1]

with open(mrp_in) as infile:
    for line in infile:
        mrp = json.loads(line)
        print(mrp['id'])
        labels = get_id2lex(mrp)
        edges = get_mrp_edges(mrp, get_remote = False)
        irtg = edge2irtg(edges, labels)
        print(irtg)
        print('_'*40)
