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
