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
import os
import sys
import re
import ast #for converting lines in mrp into a python dict

test = {"id": "reviews-086839-0002", "flavor": 1, "framework": "ucca", "version": 0.9, "time": "2019-04-11 (22:03)", "input": "One of the worst experiences I've ever had with a auto repair shop.", "tops": [16], "nodes": [{"id": 0, "anchors": [{"from": 0, "to": 3}]}, {"id": 1, "anchors": [{"from": 4, "to": 6}]}, {"id": 2, "anchors": [{"from": 7, "to": 10}]}, {"id": 3, "anchors": [{"from": 11, "to": 16}]}, {"id": 4, "anchors": [{"from": 17, "to": 28}]}, {"id": 5, "anchors": [{"from": 29, "to": 30}]}, {"id": 6, "anchors": [{"from": 30, "to": 33}]}, {"id": 7, "anchors": [{"from": 34, "to": 38}]}, {"id": 8, "anchors": [{"from": 39, "to": 42}]}, {"id": 9, "anchors": [{"from": 43, "to": 47}]}, {"id": 10, "anchors": [{"from": 48, "to": 49}]}, {"id": 11, "anchors": [{"from": 50, "to": 54}]}, {"id": 12, "anchors": [{"from": 55, "to": 61}]}, {"id": 13, "anchors": [{"from": 62, "to": 66}]}, {"id": 14, "anchors": [{"from": 66, "to": 67}]}, {"id": 15}, {"id": 16}, {"id": 17}, {"id": 18}, {"id": 19}, {"id": 20}], "edges": [{"source": 15, "target": 3, "label": "C"}, {"source": 19, "target": 20, "label": "E"}, {"source": 19, "target": 14, "label": "U"}, {"source": 18, "target": 7, "label": "T"}, {"source": 17, "target": 15, "label": "D"}, {"source": 19, "target": 9, "label": "R"}, {"source": 17, "target": 18, "label": "A"}, {"source": 18, "target": 8, "label": "P"}, {"source": 18, "target": 5, "label": "A"}, {"source": 15, "target": 1, "label": "F"}, {"source": 15, "target": 2, "label": "F"}, {"source": 20, "target": 12, "label": "P"}, {"source": 18, "target": 4, "label": "A", "properties": ["remote"], "values": [True]}, {"source": 19, "target": 11, "label": "E"}, {"source": 15, "target": 0, "label": "Q"}, {"source": 19, "target": 13, "label": "C"}, {"source": 17, "target": 4, "label": "P"}, {"source": 18, "target": 6, "label": "F"}, {"source": 17, "target": 19, "label": "A"}, {"source": 16, "target": 17, "label": "H"}, {"source": 19, "target": 10, "label": "F"}]}

def get_id2lex(mrp_dict):
    '''
    input: takes one line of the mrp corpus
    output: gets the mapping between ids and tokens
    '''
    sent = mrp_dict["input"]
    id_to_lex = {}
    #print(sent_indexed)
    for node in mrp_dict["nodes"]:
        #if node['label'] != 'Non-Terminal':
        #print(node)
        #print(id_to_lex)
        spans = []
        node_id = node["id"]
        if 'anchors' in node.keys() and node['anchors']:
            for anchor in node["anchors"]:
                spans.append(anchor["from"])
                spans.append(anchor["to"])
            begin = min(spans)
            end = max(spans)
            for i in sent:
                lex = sent[begin:end]
                # ADDED BY MM TO REVERT JUST DELETE UNTIL
                lex = re.sub(r'[–—−]', '-', lex).lower()
                if "’" in lex:
                    lex = lex.replace("’", "'")
                if "”" in lex:
                    lex = lex.replace("”", "RIGHT_QUOTATION")
                if "“" in lex:
                    lex = lex.replace("“", "LEFT_QUOTATION")
                if '"' in lex:
                    lex = lex.replace('"', "QUOTATION_MARK")
                if "…" in lex:
                    lex = lex.replace("…", "...")
                if "[" in lex:
                    lex = lex.replace("[", "LEFT_SQUARE_BRACKET")
                if "]" in lex:
                    lex = lex.replace("]", "RIGHT_SQUARE_BRACKET")
                if "ć" in lex:
                    lex = lex.replace("ć", "c_acute")
                #HERE
                id_to_lex[node_id] = lex
        else:
            id_to_lex[node_id] = "Non-Terminal"
    return id_to_lex


def get_mrp_edges(mrp_dict, get_remote = False):
    edges = {}
    for edge_dict in mrp_dict["edges"]:
        if get_remote == False:
            if "properties" not in edge_dict.keys():
                edges[(edge_dict["source"], edge_dict["target"])] = edge_dict["label"]
        else:
            edges[(edge_dict["source"], edge_dict["target"])] = edge_dict["label"]
    return edges
