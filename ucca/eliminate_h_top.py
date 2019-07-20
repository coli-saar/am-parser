from a_star_mrp import get_roots

def eliminate_h(edge_dict):
    roots = get_roots(edge_dict)
    #print(roots)
    if len(roots) == 1:
        for root in roots:
            daughter_edges = []
            for (u,v) in edge_dict.keys():
                if u == root:
                    daughter_edges.append((u,v))
            if len(daughter_edges) == 1 and edge_dict[daughter_edges[0]] == 'H' and len(edge_dict.keys()) > 1:
                del edge_dict[daughter_edges[0]]
                new_root = get_roots(edge_dict)[0]
                for (u,v) in list(edge_dict.keys()):
                    if u == new_root:
                        new_edge = (str(new_root) + '<root>', v)
                        edge_dict[new_edge] = edge_dict[(u,v)]
                        del edge_dict[(u,v)]
            else:
                for (u,v) in list(edge_dict.keys()):
                    for root in roots:
                        if u == root:
                            new_edge = (str(root) + '<root>', v)
                            edge_dict[new_edge] = edge_dict[(u,v)]
                            del edge_dict[(u,v)]
    return edge_dict

def add_h(edge_dict):
    roots = get_roots(edge_dict)
    #print(roots)
    #print(roots)
    if len(roots) == 1:
        for root in roots:
            daughter_edges = []
            for (u,v) in edge_dict.keys():
                if u == root:
                    daughter_edges.append((u,v))
            if not any([edge_dict[(u,v)] for (u, v) in daughter_edges if edge_dict[(u,v)] == 'H']):
                n = max([u for (u,v) in edge_dict.keys() if type(u) == int])
                n = n+1
                edge_dict[(n, root)] = 'H'
    return edge_dict
