import sys
import os
import xml.etree.ElementTree as et


def get_edges_from_xml(filename):
    print(filename)
    #extra lines for irtg format:
    sent = []
    #xml_file = directory + filename
    #maps words to ids
    id_label_dict = {}
    #maps ids to type, only for non-terminal units
    #dict with connected nodes (n1, n2)-label as key-value pairs
    edges = {}
    nodes = set()
    tree = et.parse(filename)
    #files have two layers, one for the terminals (layer 0), and one for the non-terminals and all
    #relations between them (layer 1)
    #for foundational layer with the actual words:
    for node in tree.getroot()[1]:
        id = node.attrib.get('ID')
        for attribute in node:
            word = attribute.attrib.get('text')
            i = attribute.attrib.get('paragraph_position')
            id_label_dict[id] = (word, i)
            sent.append(word)
            nodes.add(id)
    #for layer with Non terminals
    for node in tree.getroot()[2]:
        #print(node.tag, node.attrib)
        id = node.attrib.get('ID')
        nodes.add(id)
        type = node.attrib.get('type')
        id_label_dict[id] = type
        for edge in node:
            #print(edge.tag, edge.attrib)
            end_edge = edge.attrib.get('toID')
            edge_label = edge.attrib.get('type')
            if end_edge:
                edges[(id, end_edge)] = edge_label
            else:
                pass
    return edges, nodes, id_label_dict
