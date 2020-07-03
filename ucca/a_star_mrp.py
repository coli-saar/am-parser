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
import collections
import sys
import re
from nltk.metrics import *

test = {"id": "reviews-086839-0002", "flavor": 1, "framework": "ucca", "version": 0.9, "time": "2019-04-11 (22:03)", "input": "One of the worst experiences I've ever had with a auto repair shop.", "tops": [16], "nodes": [{"id": 0, "anchors": [{"from": 0, "to": 3}]}, {"id": 1, "anchors": [{"from": 4, "to": 6}]}, {"id": 2, "anchors": [{"from": 7, "to": 10}]}, {"id": 3, "anchors": [{"from": 11, "to": 16}]}, {"id": 4, "anchors": [{"from": 17, "to": 28}]}, {"id": 5, "anchors": [{"from": 29, "to": 30}]}, {"id": 6, "anchors": [{"from": 30, "to": 33}]}, {"id": 7, "anchors": [{"from": 34, "to": 38}]}, {"id": 8, "anchors": [{"from": 39, "to": 42}]}, {"id": 9, "anchors": [{"from": 43, "to": 47}]}, {"id": 10, "anchors": [{"from": 48, "to": 49}]}, {"id": 11, "anchors": [{"from": 50, "to": 54}]}, {"id": 12, "anchors": [{"from": 55, "to": 61}]}, {"id": 13, "anchors": [{"from": 62, "to": 66}]}, {"id": 14, "anchors": [{"from": 66, "to": 67}]}, {"id": 15}, {"id": 16}, {"id": 17}, {"id": 18}, {"id": 19}, {"id": 20}], "edges": [{"source": 15, "target": 3, "label": "C"}, {"source": 19, "target": 20, "label": "E"}, {"source": 19, "target": 14, "label": "U"}, {"source": 18, "target": 7, "label": "T"}, {"source": 17, "target": 15, "label": "D"}, {"source": 19, "target": 9, "label": "R"}, {"source": 17, "target": 18, "label": "A"}, {"source": 18, "target": 8, "label": "P"}, {"source": 18, "target": 5, "label": "A"}, {"source": 15, "target": 1, "label": "F"}, {"source": 15, "target": 2, "label": "F"}, {"source": 20, "target": 12, "label": "P"}, {"source": 18, "target": 4, "label": "A", "properties": ["remote"], "values": [True]}, {"source": 19, "target": 11, "label": "E"}, {"source": 15, "target": 0, "label": "Q"}, {"source": 19, "target": 13, "label": "C"}, {"source": 17, "target": 4, "label": "P"}, {"source": 18, "target": 6, "label": "F"}, {"source": 17, "target": 19, "label": "A"}, {"source": 16, "target": 17, "label": "H"}, {"source": 19, "target": 10, "label": "F"}]}


def get_roots(edge_dict):
    eus=set()
    evs=set()
    for (eu, ev) in list(edge_dict.keys()):
        eus.add(eu)
        evs.add(ev)
    roots = list(eus.difference(evs))
    return roots

def reconstruct_path(came_from, current):
    total_path = [current]
    #print('TOTAL PATH')
    #print(total_path)
    while current in came_from.keys():
        #print('CURRENT')
        #print(current)
        #print('NEW')
        #print('_________________________________________')
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def get_cost(edge_dict, priority_queue):
    heuristic_cost = {}
    for (eu, ev) in edge_dict.keys():
        heuristic_cost[(eu, ev)] = priority_queue[edge_dict[(eu, ev)]]
    return heuristic_cost

def get_neighbours(node, edge_dict):
    neighbours = [ev for (eu, ev) in edge_dict.keys() if eu == node]
    return neighbours

def node_to_token_index(mrp_companion, mrp_dict, id_lex_dict, mrp_sent_id):
    node_to_index = {}
    do_second_iteration = False
    for node in id_lex_dict.keys():
        if id_lex_dict[node] != 'Non-Terminal':
            for mrp_node in mrp_dict['nodes']:
                if mrp_node['id'] == node:
                    if 'anchors' in mrp_node:
                        span_of_first = str(mrp_node['anchors'][0]['from'])+':'+str(mrp_node['anchors'][0]['to'])
                        if not any([span for span in mrp_companion[mrp_sent_id]['spans'].keys() if span == span_of_first]):
                            do_second_iteration = True
                        for span in mrp_companion[mrp_sent_id]['spans'].keys():
                            if span == span_of_first:
                                node_to_index[node] = mrp_companion[mrp_sent_id]['spans'][span]
    for node in mrp_dict['nodes']:
        for anchor in node["anchors"]:
            spans.append(anchor["from"])
            spans.append(anchor["to"])
    if do_second_iteration == True:
        mrp_spans = []
        companion_spans = []
        orphans = []
        for node in mrp_dict['nodes']:
            if 'anchors' in node.keys():
                id = node['id']
                spans = []
                for anchor in node["anchors"]:
                    spans.append(anchor["from"])
                    spans.append(anchor["to"])
                begin = min(spans)
                end = max(spans)
                word =  mrp_dict['input'][begin:end]
                if '’' in word:
                    word = word.replace('’', "'")
                word = re.sub(r'[–—−]', '-', word).lower()
                mrp_token_info = (id, word, str(begin)+':'+str(end))
                mrp_spans.append(mrp_token_info)
        for span in mrp_companion[mrp_sent_id]['spans']:
            companion_token_info = (mrp_companion[mrp_sent_id]['tokenization'][mrp_companion[mrp_sent_id]['spans'][span] - 1], span)
            companion_spans.append(companion_token_info)
        mrp_spans = sorted(mrp_spans, key=lambda x:int(x[-1].split(':')[0]))
        companion_spans = sorted(companion_spans, key=lambda x:int(x[-1].split(':')[0]))
        orphans = mrp_spans
        for token in mrp_spans:
            #print(token)
            for other_token in companion_spans:
                #print(other_token)
                if token[1] == other_token[0] and token in orphans:
                    #print(token)
                    #print(other_token)
                    orphans.remove(token)
                    node_to_index[token[0]] = mrp_companion[mrp_sent_id]['spans'][other_token[-1]]
            #print('_______________________-')
        for orphan in orphans:
            #print('ORPHANS ', orphans)
            #print('ORPHAN ', orphan)
            #print('COMPANION SPANS ', companion_spans)
            possible_matches = [token for token in companion_spans if (token[0] == orphan[1] or  orphan[1] in token[0]) and abs(int(token[-1].split(':')[0]) - int(orphan[-1].split(':')[0])) < 10]
            #print('POSSIBLE MATCHES ', possible_matches)
            #print('_____________________________________')
            if len(possible_matches) > 0:
                optimal_candidate = min(possible_matches, key = lambda token:abs(int(token[-1].split(':')[0]) - int(orphan[-1].split(':')[0])))
                node_to_index[orphan[0]] = mrp_companion[mrp_sent_id]['spans'][optimal_candidate[-1]]
            else:
                multi_term_begining = [token for token in companion_spans if orphan[1].startswith(token[0]) or orphan[1].startswith(token[0][1:])  and abs(int(token[-1].split(':')[0]) - int(orphan[-1].split(':')[0])) < 10]
                #print(multi_term_begining)
                if len(multi_term_begining) > 0:
                    optimal_candidate = min(multi_term_begining, key = lambda token:abs(int(token[-1].split(':')[0]) - int(orphan[-1].split(':')[0])))
                    node_to_index[orphan[0]] = mrp_companion[mrp_sent_id]['spans'][optimal_candidate[-1]]

                else:
                    right_neighbour = [token for token in companion_spans if token[-1].split(':')[0] == orphan[-1].split(':')[0]]
                    if len(right_neighbour) > 0:
                        right_neighbour = right_neighbour[0]
                        node_to_index[orphan[0]] = mrp_companion[mrp_sent_id]['spans'][right_neighbour[-1]]
                    else:
                        left_neighbour = [token for token in companion_spans if token[-1].split(':')[0] == orphan[-1].split(':')[1]]
                        if len(left_neighbour) > 0:
                            left_neighbour = left_neighbour[0]
                            node_to_index[orphan[0]] = mrp_companion[mrp_sent_id]['spans'][left_neighbour[-1]]
                        else:
                            by_edit = [token for token in companion_spans if edit_distance(token[0], orphan[1]) < 10]
                            if by_edit:
                                optimal_by_edit = min(by_edit, key = lambda token:abs(int(token[-1].split(':')[0]) - int(orphan[-1].split(':')[0])))
                                node_to_index[orphan[0]] = mrp_companion[mrp_sent_id]['spans'][optimal_by_edit[-1]]
    return node_to_index

def a_star(start, cost_dict, label_dict):
    #closedSet
    closedSet = set()
    current = start
    #OpenSet
    openset = {start}
    #map
    camefrom = {}
    gscore = collections.defaultdict(lambda:100)
    gscore[start] = 0
    fscore = collections.defaultdict(lambda:100)
    fscore[start] = 0
    while len(openset) != 0:
        #print('OPENSET FROM A*')
        #print(openset)
        open_scored = {node:fscore[node] for node in list(openset)}
        current = min(open_scored, key = open_scored.get)
        #print('came from')
        #print(camefrom)
        if current and type(label_dict[current]) == str and label_dict[current] != 'Non-Terminal':
            #print('CURRENT BEING FED TO reconstruct_path')
            #print(current)
            return reconstruct_path(camefrom, current)
        closedSet.add(current)
        openset.remove(current)
        current_neighbours = get_neighbours(current, cost_dict)
        for neighbour in current_neighbours:
            if neighbour in closedSet:
                continue
            tentative_gScore = gscore[current] + cost_dict[(current, neighbour)]
            if neighbour not in openset:
                openset.add(neighbour)
            if tentative_gScore >= gscore[neighbour]:
                continue
            camefrom[neighbour] = current
            gscore[neighbour] = tentative_gScore
            fscore[neighbour] = gscore[neighbour] + cost_dict[(current, neighbour)]


def align(edge_dict, priority_dict, mrp_dict, node_to_token_index, label_dict):
    #print(label_dict)
    alignments = ''
    costs = get_cost(edge_dict, priority_dict)
    visited = set()
    while any([v for (u, v) in costs.keys() if type(label_dict[v]) == str and label_dict[v]!='Non-Terminal']):
        roots = get_roots(costs)
        for root in roots:
            path = a_star(root, costs, label_dict)
            print(path)
            if path != None:
                edges_to_remove = list(zip(path, path[1:]))
                if len(path) > 1:
                    edges_to_remove += [(eu, ev) for (eu, ev) in costs.keys() if eu == root and (eu, ev) not in edges_to_remove and label_dict[ev] == 'Non-Terminal']
                #write alignment format
                    for node in path:
                        if label_dict[node]!='Non-Terminal':
                            alignments += str(node) + '!' + '||' + str(node_to_token_index[node]) + '||' + '1.0 '
                        else:
                            alignments += str(node) + '|'
                    for edge in edges_to_remove:
                        del costs[edge]
                elif len(path) == 1:
                    #print('LENGTH OF PATH IS 1')
                    edges_to_remove = [(eu, ev) for (eu, ev) in costs.keys() if eu == root and (eu, ev) not in edges_to_remove and label_dict[ev] != 'Non-Terminal']
                    #print('EDGES TO REMOVE')
                    #print(edges_to_remove)
                    for (eu, ev) in edges_to_remove:
                        if type(label_dict[ev]) == str and label_dict[ev] != 'Non-Terminal':
                            #print("YES")
                            alignments += str(eu) + '|'+str(ev) + '!' + '||' + str(node_to_token_index[ev]) + '||' + '1.0 '
                    for edge in edges_to_remove:
                        del costs[edge]
            elif not path:
                #print('COULDNT FIND PATH')
                #print(path)
                edges_to_remove = []
                for (eu, ev) in costs.keys():
                    if eu == root and type(label_dict[ev]) == str and label_dict[ev] != 'Non-Terminal':
                        edges_to_remove.append((eu,ev))
                for (eu, ev) in edges_to_remove:
                    alignments += str(eu) + '|'+str(ev) + '!' + '||' + str(node_to_token_index[ev]) + '||' + '1.0 '
                    del costs[(eu, ev)]

        #print(costs)
        #print(alignments)
        #print('______________________________________________________________')
    return alignments
