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
