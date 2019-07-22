from get_edges_from_mrp import test, get_id2lex, get_mrp_edges
from edge_to_irtg import edge2irtg
from nltk.tokenize.moses import MosesDetokenizer

#TODO: maybe implement a node/edge class and add methods for getting sisters, daughter and parent


def lower_edge(edge_dict):
    '''
    Input:
    edge_dict: dictionary of edges (graph)
    edge_label_to_lower: the edges that we want to lower based on their label
    priority_where_to_lower: a list sorted according to along which edges we want to lower them [higher priority.... lower priority]
    '''
    to_lower = []
    for (u, v) in list(edge_dict.keys()):
        if '_' in edge_dict[(u,v)]:
            to_lower.append((u, v))
    for (u, v) in to_lower:
        for (s, t) in list(edge_dict.keys()):
            if s==u:
                if '-' in edge_dict[(s,t)]:
                    if edge_dict[(s,t)].split('-')[0] == edge_dict[(u,v)].split('_')[1]:
                        edge_dict[(t, v)] = edge_dict[(u,v)][0]
                        del edge_dict[(u, v)]
    for (u,v) in edge_dict.keys():
        edge_dict[(u,v)] = edge_dict[(u,v)][0]
    return edge_dict

def raise_edge(edge_dict, edge_label_to_raise, priority_where_to_raise, mark = True, seek_marks = False, label_dict):
        '''
        Input:
        edge_dict: dictionary of edges (graph)
        edge_label_to_lower: the edges that we want to raise based on their label
        priority_where_to_lower: a list sorted according to along which edges we want to raise them [higher priority.... lower priority]
        '''
        to_raise = []
        for (u, v) in list(edge_dict.keys()):
            if edge_dict[(u,v)] == edge_label_to_raise and label_dict[u] != 'Non-Terminal':
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
                                if '-l' in edge_dict[(s, t)]:
                                    edge_dict[(s, t)] = edge_dict[(s, t)][0]
                                    break
                                else:
                                    edge_dict[(s, t)] = edge_dict[(s, t)] +'-r'
                                    break
            else:
                for label in priority_where_to_raise:
                    for (s, t) in list(parents.keys()):
                        if t == u:
                            if parents[(s, t)] == label[0]:
                                edge_dict[(s, v)] = edge_dict[(u, v)] + '_'+label
                                del edge_dict[(u,v)]
                                if mark == True:
                                    if '-l' in edge_dict[(s, t)]:
                                        edge_dict[(s, t)] = edge_dict[(s, t)][0]
                                        break
                                    else:
                                        edge_dict[(s, t)] = edge_dict[(s, t)] + "-r"
                                        break
                break
        return edge_dict
