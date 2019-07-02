import json
import sys

corpus = sys.argv[1]
outfile = sys.argv[2]

with open(corpus) as infile:
    added = []
    for line in infile:
        mrp = json.loads(line)
        for node in mrp['nodes']:
            if 'anchors' in node.keys():
                if len(node['anchors']) > 1:
                    check_contiguous = zip(node['anchors'], node['anchors'][1:])
                    for node_1, node_2 in check_contiguous:
                        #print(node_1, node2)
                        if node_1['to'] < node_2['from'] - 2:
                            print(mrp['id'])
                            print(node_1, node_2)
                            with open(outfile, 'a') as out:
                                if mrp not in added:
                                    #out.write(json.dumps(mrp)+'\n')
                                    #added.append(mrp)
                                    break
        #break
