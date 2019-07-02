from get_edges_from_mrp import test, get_id2lex, get_mrp_edges
from edge_to_irtg import edge2irtg
from nltk.tokenize.moses import MosesDetokenizer

#TODO: maybe implement a node/edge class and add methods for getting sisters, daughter and parent


def lower_edge(edge_dict, edge_label_to_lower, priority_where_to_lower, mark = False):
    '''
    Input:
    edge_dict: dictionary of edges (graph)
    edge_label_to_lower: the edges that we want to lower based on their label
    priority_where_to_lower: a list sorted according to along which edges we want to lower them [higher priority.... lower priority]
    '''
    to_lower = []
    for (u, v) in list(edge_dict.keys()):
        if edge_dict[(u,v)] == edge_label_to_lower:
            to_lower.append((u, v))
    for (u, v) in to_lower:
        #check if it's already attached next to a C node (compressed or otherwise)
        sisters = {}
        for (s, t) in list(edge_dict.keys()):
            if s == u:
                sisters[(s, t)] = edge_dict[(s, t)]
        if ("Q" in sisters.values() or "E" in sisters.values() or "F" in sisters.values() or "R" in sisters.values()): #and not ("P" in sisters.values() or "S" in sisters.values()):
            pass
            #if the edge has sister edges with the above labels, then it attaches to a C, and therefore cannot be lowered any more
        else:
            for label in priority_where_to_lower:
                for (q, r) in list(sisters.keys()):
                    if sisters[(q, r)] == label:
                        edge_dict[(r, v)] = edge_dict[(u, v)]
                        del edge_dict[(u, v)]
                        if mark == True:
                            edge_dict[(q, r)] = edge_dict[(q, r)] + "-lowered_thru"
    return edge_dict

def raise_edge(edge_dict, edge_label_to_raise, priority_where_to_raise, mark = False, seek_marks = False):
        '''
        Input:
        edge_dict: dictionary of edges (graph)
        edge_label_to_lower: the edges that we want to raise based on their label
        priority_where_to_lower: a list sorted according to along which edges we want to raise them [higher priority.... lower priority]
        '''
        to_raise = []
        for (u, v) in list(edge_dict.keys()):
            if edge_dict[(u,v)] == edge_label_to_raise:
                to_raise.append((u, v))
        for (u, v) in to_raise:
            parents = {}
            for (s, t) in list(edge_dict.keys()):
                if t == u:
                    parents[(s, t)] = edge_dict[(s, t)]
            if seek_marks == True:
                for (s, t) in list(parents.keys()):
                    if t == u:
                        if '-' in parents[(s, t)]:
                            edge_dict[(s, v)] = edge_dict[(u, v)]
                            del edge_dict[(u,v)]
                            if mark == True:
                                if 'lowered_thru' in edge_dict[(s, t)]:
                                    edge_dict[(s, t)] = edge_dict[(s, t)][0]
                                    break
                                else:
                                    edge_dict[(s, t)] = edge_dict[(s, t)] +'-raised_thru'
                                    break
            else:
                for label in priority_where_to_raise:
                    for (s, t) in list(parents.keys()):
                        if t == u:
                            if parents[(s, t)] == label[0]:
                                edge_dict[(s, v)] = edge_dict[(u, v)]
                                del edge_dict[(u,v)]
                                if mark == True:
                                    if 'lowered_thru' in edge_dict[(s, t)]:
                                        edge_dict[(s, t)] = edge_dict[(s, t)][0]
                                        break
                                    else:
                                        edge_dict[(s, t)] = edge_dict[(s, t)] + "-raised_thru"
                                        break
                break
        return edge_dict
