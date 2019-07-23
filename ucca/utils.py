def number_edges(edge_dict, label):
    '''
    takes an edge dict and enumerates all edges with a given label that have the same parent node
    '''
    for (u,v) in edge_dict.keys():
        children = []
        for (s, t) in edge_dict.keys():
            if s == u:
                if edge_dict[(s, t)] == label:
                    children.append((s,t))
        counter = 1
        for (s, t) in children:
            edge_dict[(s, t)] = edge_dict[(s,t)] + str(counter)
            counter += 1
    return edge_dict
