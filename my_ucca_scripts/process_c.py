import sys
import os

from get_edges_from_mrp import get_mrp_edges


def compress_c_edge(edge_dict):
    '''
    input: dictionary of unprocessed edges in the format (source_node, target_node): label
    output: turns the nodeid into the node with the center edge and then flips all nodes around it
    however, it takes into account chained C's e.g. u -c-> v, v-c->w and doesn't try to compress those
    as well as coordination examples
    in the mappings dict, keep track of changes made to later be able to undo them. Format is
    {is_mapped_to_after_compression: was_mapped_to_before_compression}
    '''
    c_edges = []
    mappings = {}
    for (u, v) in edge_dict.keys():
        if edge_dict[(u, v)] == 'C':
            c_edges.append((u, v))
    total_cs = len(c_edges)
    for (u, v) in c_edges:
        print(c_edges)
        for (s, t) in edge_dict.keys():
            if s == v and edge_dict[(s, t)] == "C":
                if (u,v) in c_edges:
                    c_edges.remove((u,v))
    visited = {}
    c_edges = []
    for (u, v) in c_edges:
        if not u in visited:
            visited.add(u)
            c_edges.append((u,v))
    compressable_cs = len(c_edges)
    for (u, v) in c_edges:
        for (s, t) in list(edge_dict.keys()):
            if t == u:
                edge_dict[(s, v)] = edge_dict[(s, t)]
                del edge_dict[(s,t)]
                mappings[v] = t
        for (s, t) in list(edge_dict.keys()):
            if s== u:
                edge_dict[(v, t)] = edge_dict[(s, t)]
                del edge_dict[(s,t)]
    for (u, v) in list(edge_dict.keys()):
        if u == v:
            del edge_dict[(u, v)]
    return total_cs, compressable_cs, edge_dict#, mappings

def decompress_c(edge_dict):
    '''
    decompresses the graphs by looking for a non-terminal with tell-tale outgoing edges
    that cannot exist without a center and then takes the source node as the center
    RETURNS
    _________
    uncompressed dicts, with similar surface structure as the original uncompressed edge dict
    but instead of the original names of the nodes, we use NONTERMINAL + counter as a node id
    '''
    counter = 0
    for (u, v) in list(edge_dict.keys()):
        daughters = []
        for (s, t) in list(edge_dict.keys()):
            if s == v:
                daughters.append(edge_dict[(s, t)])
        if ("Q" in daughters or "E" in daughters or "F" in daughters or "R" in daughters) and not ("C" in daughters or "P" in daughters or "S" in daughters):
            compressed = v
            #if mappings == None:
            for (q, r) in list(edge_dict.keys()):
                if q == compressed:
                    edge_dict[("NONTERMINAL" + str(counter), r)] = edge_dict[q, r]
                    del edge_dict[(q, r)]
                    edge_dict[("NONTERMINAL"+str(counter), compressed)] = 'C'
                if r == compressed:
                    edge_dict[(q, "NONTERMINAL" + str(counter))] = edge_dict[q, r]
                    del edge_dict[(q, r)]
                    edge_dict[("NONTERMINAL"+str(counter), compressed)] = 'C'
                #TODO: use mappings to rename the nonterminal nodes?
        counter += 1
    return edge_dict
