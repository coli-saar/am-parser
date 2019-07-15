
test = [('b','c'),('a','b'), ('d','e'), ('e','f')]

def chain_cs(edge_list):
    edge_list = edge_list
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

print(chain_cs(test))
