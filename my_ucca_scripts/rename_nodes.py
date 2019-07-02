from get_edges_from_mrp import get_mrp_edges,get_id2lex
import sys


def rename_nodes(edge_dict_processed, edge_dict_gold, label_dict_processed, label_dict_gold):
    to_rename = []
    unused_gold_nodes = []
    processed_us = [u for (u, v) in edge_dict_processed.keys()]
    processed_vs = [v for (u, v) in edge_dict_processed.keys()]
    for identifier in list(label_dict_processed.keys()):
        if identifier.startswith("NONTERMINAL"):
            to_rename.append(identifier)
    for identifier in list(label_dict_gold.keys()):
        if identifier not in us and idenfier not in vs:
            unused_gold_nodes.append(identifier)
    to_rename_match_scores = {}
    for node in unused_gold_nodes:
