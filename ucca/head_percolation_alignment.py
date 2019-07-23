import copy

def percolate(edge_dict, priority_queue, label_dict):
    copy_edge_dict = copy.copy(edge_dict)
    alignment_dict = {v:[] for (u,v) in copy_edge_dict.keys() if label_dict[v] != 'Non-Terminal'}
    while len(copy_edge_dict.keys()) > 0:
        for category in priority_queue:
            for u, v in list(copy_edge_dict.keys()):
                if label_dict[u] != 'Non-Terminal' and label_dict[v] != 'Non-Terminal':
                    del copy_edge_dict[(u,v)]
            for u, v in list(copy_edge_dict.keys()):
                if (u, v) in copy_edge_dict.keys():
                    if label_dict[v] != 'Non-Terminal' and copy_edge_dict[(u,v)][0] == category[0]:
                        alignment_dict[v].append(u)
                        del copy_edge_dict[(u, v)]
                        for s, t in list(copy_edge_dict.keys()):
                            if s == u and t != v:
                                del copy_edge_dict[(s,t)]
                            if t == u:
                                copy_edge_dict[(s,v)] = copy_edge_dict[(s,t)]
                                del copy_edge_dict[(s, t)]
    #print(alignment_dict)
    return alignment_dict
