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
from get_edges_from_mrp import test, get_id2lex, get_mrp_edges
import re

'''
def get_input(edge_dict, label_dict):
    input = []
    terminal_ids = [id for id in label_dict.keys() if type(id) == int]
    for node in sorted(terminal_ids):
        if label_dict[node] != "Non-Terminal" and not label_dict[node].startswith("NONTERMINAL"):
        #if type(label_dict[node]) != int:
            input.append(label_dict[node])
            #print(input)
        else:
            pass
    input_tokenized = detokenizer.detokenize(input, return_str = True)
    return input_tokenized
'''
def get_nodes(label_dict, input):
    nodes = []
    for node in label_dict.keys():
        if label_dict[node] in input.lower().split():
            if ' ' not in label_dict[node]:
                node_anchor_dict = {}
                span_regex = re.compile(re.escape(label_dict[node]))
                span_match = span_regex.search(input)
                (begin, end) = span_match.span()
                node_anchor_dict = {'id': node, 'anchors' :[{'from': begin, 'to':end}]}
                nodes.append(node_anchor_dict)
            else:
                node_anchor_dict = {}
                multi_word_exp = label_dict[node].split()
                node_anchor_dict['id'] = node
                node_anchor_dict['anchors'] = []
                for word in multi_word_exp:
                    span_regex = re.compile(re.escape(label_dict[node]))
                    span_match = span_regex.search(input)
                    (begin, end) = span_match.span()
                    from_begin = {'from':begin, 'to':end}
                    node_anchor_dict['anchors'].append(from_begin)
                    nodes.append(node_anchor_dict)
        else:
            node_anchor_dict = {}
            node_anchor_dict['id']=node
            nodes.append(node_anchor_dict)
    return nodes

def get_tops(edge_dict):
    us=set()
    vs=set()
    for (u, v) in edge_dict.keys():
        us.add(u)
        vs.add(v)
    tops = list(us.difference(vs))
    return tops


def get_edges(edge_dict):
    edges = []
    for (u, v) in edge_dict.keys():
        mrp_edges = {'source': u, 'target':v, 'label': edge_dict[(u,v)]}
        edges.append(mrp_edges)
    return edges



def irtg2mrp(edge_dict, label_dict):
    input = get_input(edge_dict, label_dict)
    nodes = get_nodes(label_dict, input)
    edges = get_edges(edge_dict)
    tops = get_tops(edge_dict)
    mrp = {}
    mrp['input'] = input
    mrp['nodes'] = nodes
    mrp['edges'] = edges
    mrp['tops'] = tops
    return mrp
