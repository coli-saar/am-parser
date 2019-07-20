import json
import sys
from get_edges_from_mrp import get_id2lex, get_mrp_edges


corpus = sys.argv[1]

with open(corpus) as infile:
    for line in infile:
        mrp = json.loads(line)
        labels = get_id2lex(mrp)
        edges = get_mrp_edges(mrp)
        nodes = set()
        node_mentions_in_edges = set()
        for node_id in labels.keys():
            nodes.add(node_id)
        for (u, v) in edges.keys():
            node_mentions_in_edges.add(u)
            node_mentions_in_edges.add(v)
