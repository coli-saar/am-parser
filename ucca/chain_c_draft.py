#
# Copyright (c) 2020 Saarland University.
#
# This file is part of AM Parser
# (see https://github.com/coli-saar/am-parser/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
[10/Bryan-Michael Cox -U-> 11/-; 30/Non-Terminal -C-> 29/Non-Terminal; 29 -D-> 16/new; 24/Non-Terminal -P-> 26/Non-Terminal; 21/Non-Terminal -A-> 1/collaborators; 24 -A-> 25/Non-Terminal; 29 -A-> 17/collaborators; 28/Non-Terminal -N-> 12/and; 25 -U-> 14/,; 28 -C-> 13/Randy Jackson; 28 -C-> 6/Jermaine Dupri; 21 -T-> 0/Longtime; 25 -N-> 15/as well as; 27/Non-Terminal -C-> 28; 30 -E-> 31/Non-Terminal; 28 -C-> 8/Johntá Austin; 25 -C-> 30; 28 -U-> 9/,; 31 -R-> 18/such as; 31 -C-> 19/Marc Shaiman; 23/Non-Terminal -H-> 24; 26 -R-> 2/for; 28 -U-> 7/,; 22/Non-Terminal -E-> 27; 26 -C-> 4/project; 27 -R-> 5/included; 22 -C-> 21; 28 -C-> 10; 25 -C-> 22; 26 -E-> 3/the; 31 -U-> 20/.]





def compress_chained(edge_dict, chained_c_edges, coordinated_c_edges):
    '''
    main idea is to check if edge is part of a coordination, and then compressing if it's not
    '''
    #for chain in chained_c_edges:
    for (u,v) in chained_c_edges:
        for (s,t) in chained_c_edges:
            for edge in edge_dict.keys():
                if t == u and (u,v) not in coordinated_c_edges : #then (s,t) is the parent of (u,v))), bottom up compression
                        if type(u) != tuple and type(v) != tuple:
                            new_u = (u, v)
                        elif type(u) != tuple and type(v) == tuple: #we're compressing bottom up, so this should be fine, if not first place to # DEBUG
                             new_u = (u,) + v
                        for edge in list(edge_dict.keys()):
                            for (q,r) in edge:
                                if q == u:
                                    edge_dict[(edge[0], new_u)] = edge_dict[edge]
                                    del edge_dict[edge]
                        for edge in chained_c_edges:
                            for node in edge:
                                if node == u:
                                    chained_c_edges.remove(edge)      #if we don't rewrite, we can keep iterating
                                    chained_c_edges.append(new_edge)  #because now this edge is new_u
                        for edge in coordinated_c_edges:
                            for node in edge:
                                if node == u:
                                    edge = list(edge)
                                    edge[1] = new_u #defined in previous loop
                                    edge = tuple(edge)

                                    coordinated_c_edges
    return edge_dict



def compress_chained_cs(edge_dict, chained_c_edges, coordinated_c_edges):
    keep = [(u, v) for (u,v) in chained_c_edges if (u,v) in coordinated_c_edges and len([(s, t) for (s, t) in chained_c_edges if s == v])]
    for edge in chained_c_edges:
        if edge in coordinated_c_edges:
            if edge not in keep:
                chained_c_edges.remove(edge)
    while any((e[0], e[1]) for e in chained_c_edges if type(e[0]) == int or type(e[-1]) == int):
        for chain in chained_c_edges:
            if chain not in coordinated_c_edges and type(chain) != list:
                for other_chain in chained_c_edges:
                    if other_chain not in coordinated_c_edges and type(other_chain) != list:
                        if chain[-1] == other_chain[0] and chain != other_chain:
                            new_chain = chain + other_chain[1:]
                            chained_c_edges.append(new_chain)
                            chained_c_edges.remove(chain)
                            chained_c_edges.remove(other_chain)
        for other_chain in chained_c_edges:
            if other_chain not in coordinated_c_edges and type(other_chain) != list:
                if chain[-1] == other_chain[0] and chain!= other_chain:
                    new_chain = (chain[0],)+ (other_chain,)
                    chained_c_edges.append(new_chain)
                    if chain in chained_c_edges:
                        chained_c_edges.remove(chain)
                    if other_chain in chained_c_edges:
                        chained_c_edges.remove(other_chain)
        for chain in chained_c_edges:
            for link in chain:
                for edge in list(edge_dict.keys()):
                    for i, node in enumerate(edge):
                        if node == link:
                            edge = list(edge)
                            edge[i] = tuple(chain)
    for (u,v) in list(edge_dict.keys()):
        if u == v:
            del edge_dict[(u,v)]
return edge_dict
