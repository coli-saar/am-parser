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
import json
import sys

def get_mrp(id, flavor, framework, version, time, input, spans, edges):
    used_nodes = []
    nodes = []
    mrp_edges = []
    mrp = {}
    for span in spans.keys():
        node_dict = {'id' : spans[span], 'anchors':[{'from':int(span.split(':')[0]), 'to':int(span.split(':')[1])}]}
        nodes.append(node_dict)
        used_nodes.append(spans[span])
    for (u, v) in edges.keys():
        if not u in used_nodes:
            node_dict = {'id':u}
            used_nodes.append(node_dict)
        if not v in used_nodes:
            node_dict = {'id':v}
            used_nodes.append(node_dict)
        mrp_edges.append({'source':u, 'target':v, 'label':edges[(u,v)]})
    mrp['id'] = id
    mrp['flavor'] = flavor
    mrp['framework'] = framework
    mrp['version'] = version
    mrp['time'] = time
    mrp['input'] = input
    mrp['nodes'] = nodes
    mrp['edges'] = mrp_edges
    return mrp
