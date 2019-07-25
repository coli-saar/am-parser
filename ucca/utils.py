import collections
import sys
import re
from nltk.metrics import *

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


'''def node_to_token_index_mod(mrp_companion, mrp_dict, id_lex_dict, mrp_sent_id):
    node_to_index = {}
    do_second_iteration = False
    for node in id_lex_dict.keys():
        if id_lex_dict[node] != 'Non-Terminal':
            for mrp_node in mrp_dict['nodes']:
                if mrp_node['id'] == node:
                    if 'anchors' in mrp_node:
                        span_of_first = str(mrp_node['anchors'][0]['from'])+':'+str(mrp_node['anchors'][-1]['to'])
                        if not any([span for span in mrp_companion[mrp_sent_id]['spans'].keys() if span == span_of_first]): #PROBLEM HERE
                            do_second_iteration = True
                        for span in mrp_companion[mrp_sent_id]['spans'].keys():
                            if span == span_of_first:
                                node_to_index[node] = mrp_companion[mrp_sent_id]['spans'][span]
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
            for other_token in companion_spans:
                if token[1] == other_token[0] and token in orphans:
                    orphans.remove(token)
                    node_to_index[token[0]] = mrp_companion[mrp_sent_id]['spans'][other_token[-1]]
        for orphan in orphans:
            possible_matches = [token for token in companion_spans if (token[0] == orphan[1] or  orphan[1] in token[0]) and abs(int(token[-1].split(':')[0]) - int(orphan[-1].split(':')[0])) < 10]
            if len(possible_matches) > 0:
                optimal_candidate = min(possible_matches, key = lambda token:abs(int(token[-1].split(':')[0]) - int(orphan[-1].split(':')[0])))
                node_to_index[orphan[0]] = mrp_companion[mrp_sent_id]['spans'][optimal_candidate[-1]]
            else:
                multi_term_begining = [token for token in companion_spans if orphan[1].startswith(token[0]) or orphan[1].startswith(token[0][1:])  and abs(int(token[-1].split(':')[0]) - int(orphan[-1].split(':')[0])) < 10]
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
    return node_to_index'''

def node_to_token_index_mod(mrp_companion, mrp_dict, id_lex_dict, mrp_sent_id):
    node_to_index = {}
    #print(mrp_companion[mrp_sent_id]['spans'])
    #print(mrp_sent_id)
    #print(mrp_dict['nodes'])
    for node in mrp_dict['nodes']:
        #print(node_to_index)
        if 'anchors' in node.keys():
            id = node['id']
            spans = []
            for anchor in node["anchors"]:
                spans.append(anchor["from"])
                spans.append(anchor["to"])
            begin = min(spans)
            end = max(spans)
            word =  mrp_dict['input'][begin:end]
            mrp_token_info = (id, word, str(begin)+':'+str(end))
            for span in mrp_companion[mrp_sent_id]['spans'].keys():
                if span == str(begin) + ':' + str(end): #exact span match
                    node_to_index[id] = mrp_companion[mrp_sent_id]['spans'][span]
                elif int(span.split(':')[0]) > begin and int(span.split(':')[-1]) <= end: #not the first term of a multi-term-word, ignore
                    pass
                elif int(span.split(':')[0]) == begin and int(span.split(':')[-1]) < end: # first word, align to node
                    node_to_index[id] = mrp_companion[mrp_sent_id]['spans'][span]
    return node_to_index
