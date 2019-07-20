import sys
import os

from get_edges_from_mrp import get_mrp_edges


def compress_chained_cs(edge_dict, chained_c_edges, coordinated_c_edges):
    for branch in chained_c_edges:
        for chain in branch:
            for other_chain in branch:
                if chain[-1] == other_chain[0] and chain != other_chain:
                    new_chain = chain + other_chain[1:]
                    branch.append(new_chain)
                    branch.remove(chain)
                    branch.remove(other_chain)
        for chain in branch:
            if len(chain) == 2:
                turn_from = chain[-1]
                turn_to = int(str(chain[0]) + '1111')
                edge_dict[(turn_to, turn_to)] = edge_dict[(chain)]
                del edge_dict[(chain)]
                for edge in list(edge_dict.keys()):
                    #will throw bug if one node is shared by 2 C edges. Fix maybe in compress_c by removing altogether?
                    for i,node in enumerate(edge):
                        if node in chain:
                            new_edge = list(edge)
                            new_edge[i] = turn_to
                            new_edge = tuple(new_edge)
                            edge_dict[new_edge] = edge_dict[edge]
                            del edge_dict[edge]
            elif len(chain) > 2:
                zipped = list(zip(chain, chain[1:]))
                for edge in list(edge_dict):
                    if edge in zipped:
                        del edge_dict[edge]
                    else:
                        for i, node in enumerate(edge):
                            if node in chain:
                                new_edge = list(edge)
                                new_edge[i] = chain
                                new_edge = tuple(new_edge)
                                edge_dict[new_edge] = edge_dict[edge]
                                del edge_dict[edge]
    for (u,v) in list(edge_dict.keys()):
        if str(u) == str(v):
            del edge_dict[(u,v)]
    return edge_dict


def chain_cs(edge_list):
    while any([element for element in edge_list if type(element) == tuple]):
        for edge in edge_list:
            if type(edge) == tuple:
                (u, v) = edge
                chains = []
                for element in edge_list:
                    if type(element) == list:
                        for (s, t) in element:
                            if s == u or s == v or t == u or t == v:
                                for (s, t) in element:
                                    chains.append((s,t))
                                edge_list.remove(element)
                if len(chains) == 0:
                    edge_list.append([(u,v)])
                    edge_list.remove((u,v))
                else:
                    chains.append((u,v))
                    edge_list.remove((u,v))
                    edge_list.append(chains)
    else:
        return edge_list

def compress_c_edge(edge_dict):
    '''
    input: dictionary of unprocessed edges in the format (source_node, target_node): label
    output: turns the nodeid into the node with the center edge and then flips all nodes around it
    however, it takes into account chained C's e.g. u -c-> v, v-c->w and doesn't try to compress those
    as well as coordination examples
    in the mappings dict, keep track of changes made to later be able to undo them. Format is
    {is_mapped_to_after_compression: was_mapped_to_before_compression}
    '''
    vanilla_c_edges = []
    chained_c_edges = []
    shared_cs = []
    coordinated_c_edges = []
    for (u, v) in edge_dict.keys():
        if edge_dict[(u, v)] == 'C':
            vanilla_c_edges.append((u, v))
    for (u, v) in vanilla_c_edges:
        for (s, t) in edge_dict.keys():
            #chained
            if (s == v or t ==u) and edge_dict[(s, t)] == "C":
                    if (u, v) not in chained_c_edges:
                        chained_c_edges.append((u,v))
                    if (s, t) not in chained_c_edges:
                        chained_c_edges.append((s,t))
            #coordination
            if s == u and t!=v and (edge_dict[(s, t)] =='N' or edge_dict[(s, t)] =='C'):
                    if (u, v) not in coordinated_c_edges:
                        coordinated_c_edges.append((u,v))
                    if (s, t) not in coordinated_c_edges:
                        coordinated_c_edges.append((s,t))
            #shared C's just skip for now
            if v == t and (edge_dict[(s,t)] =='C' and edge_dict[(u,v)] =='C' and u!= s):
                shared_cs.append((s,t))
                shared_cs.append((u,v))
    for (u,v) in shared_cs:
        if (u,v) in vanilla_c_edges:
            vanilla_c_edges.remove((u,v))
        if (u,v) in chained_c_edges:
            chained_c_edges.remove((u,v))
    chained_c_edges = list(set(chained_c_edges).difference(set(coordinated_c_edges)))
    vanilla_c_edges = list(set(vanilla_c_edges).difference(set(chained_c_edges).union(set(coordinated_c_edges))))
    for (u, v) in vanilla_c_edges:
        for (s, t) in list(edge_dict.keys()):
            if t == u:
                edge_dict[(s, v)] = edge_dict[(s, t)]
                del edge_dict[(s,t)]
        for (s, t) in list(edge_dict.keys()):
            if s== u:
                edge_dict[(v, t)] = edge_dict[(s, t)]
                del edge_dict[(s,t)]
    for (u, v) in list(edge_dict.keys()):
        if u == v:
            del edge_dict[(u, v)]
    chained_c_edges = chain_cs(chained_c_edges)
    edge_dict = compress_chained_cs(edge_dict, chained_c_edges, coordinated_c_edges)
    return edge_dict

def decompress_c(edge_dict, label_dict):
    '''
    decompresses the graphs by looking for a non-terminal with tell-tale outgoing edges
    that cannot exist without a center and then takes the source node as the center
    RETURNS
    _________
    uncompressed dicts, with similar surface structure as the original uncompressed edge dict
    but instead of the original names of the nodes, we use NONTERMINAL + counter as a node id
    '''
    #n = max([i for i in label_dict.keys() if type(i) == int and label_dict[i] == 'Non-Terminal'])
    print(label_dict)
    labels = [label for label in label_dict.keys()]
    if len(labels) > 0:
        n = max([i for i in label_dict.keys()])
    else:
        n = 1
    contracted = []
    for (u,v) in list(edge_dict.keys()):
        if label_dict[u] != 'Non-Terminal':
            if u not in contracted:
                contracted.append(u)
    for node in contracted:
        n += 1
        if type(node) != tuple:
            for (u, v) in list(edge_dict.keys()):
                if v == node:
                    edge_dict[u, n] = edge_dict[(u, v)]
                    del edge_dict[(u, v)]
                    edge_dict[(n, v)] = 'C'
                elif u == node:
                    edge_dict[(n, v)] = edge_dict[(u, v)]
                    del edge_dict[(u, v)]
                    #IF BUG COMMENT OUT NEXT LINE
                    edge_dict[(n, node)] = 'C'
        elif type(v) == tuple:
            uncontracted = list(zip(v, v[1:]))
            edge_dict[(u, v[0])] = edge_dict[(u,v)]
            del edge_dict[(u,v)]
            for (s, t) in list(edge_dict.keys()):
                if t == v and u != s:
                    edge_dict[(s, t[0])] = edge_dict[(s,t)]
                    del edge_dict[(s,t)]
            for edge in uncontracted:
                edge_dict[edge] = 'C'
    return edge_dict
