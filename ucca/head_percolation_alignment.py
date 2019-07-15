

def percolate(edge_dict, priority_queue, label_dict):
    alignment_dict = {v:[] for (u,v) in edge_dict.keys() if label_dict[v] != 'Non-Terminal'}
    while len(edge_dict.keys()) > 0:
        print(edge_dict.keys())
        for category in priority_queue:
            for u, v in list(edge_dict.keys()):
                if label_dict[u] != 'Non-Terminal' and label_dict[v] != 'Non-Terminal':
                    del edge_dict[(u,v)]
            for u, v in list(edge_dict.keys()):
                if (u, v) in edge_dict.keys():
                    if label_dict[v] != 'Non-Terminal' and edge_dict[(u,v)] == category:
                        alignment_dict[v].append(u)
                        del edge_dict[(u, v)]
                        for s, t in list(edge_dict.keys()):
                            if s == u and t != v:
                                del edge_dict[(s,t)]
                            if t == u:
                                edge_dict[(s,v)] = edge_dict[(s,t)]
                                del edge_dict[(s, t)]
                        print(alignment_dict)
                        print('-'*40)
    return alignment_dict
