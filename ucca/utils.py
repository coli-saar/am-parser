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
