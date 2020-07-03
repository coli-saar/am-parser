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
