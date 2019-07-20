import sys
import json
import collections
import os
import random

from edge_to_irtg import edge2irtg
from get_edges_from_mrp import get_id2lex, get_mrp_edges
from convert_irtg_to_mrp import get_edges, get_mrp_edges, get_nodes, get_tops, irtg2mrp
from eliminate_h_top import eliminate_h
from a_star_mrp import *
from process_c import *
from head_percolation_alignment import percolate

mrp_dir = sys.argv[1]
tokenized_dir = sys.argv[2]
outdir = sys.argv[3]

priority_queue = 'L LR LA H P S A N C D T E R F G Q U'.split()
#priority_dict = {label:index for (index, label) in enumerate(labels)}
non_deducible = ["id", "flavour", "framework", "version", "time"]

def update_id_labels(edge_dict, label_dict):
    for (u,v) in edge_dict.keys():
        if type(u) == str:
            if '<root>' in u:
                label_dict[u] = 'Non-Terminal'
        elif u - 1111 >= 0:
            if int(str(u)[:-4]) in label_dict.keys():
                label_dict[u] = label_dict[int(str(u)[:-4])]
            else: label_dict[u] = 'Non-Terminal'
    nodes_in_edge_dict = list(set([node for edge in edge_dict.keys() for node in edge]))
    label_dict_nodes = list(label_dict.keys())
    for edge in edge_dict.keys():
        for node in edge:
            if node not in label_dict.keys():
                label_dict[node] = 'Non-Terminal'
    return label_dict




header = """###IRTG unannotated corpus file, v1.0
###
### can add comment here
###interpretation id: de.up.ling.irtg.algebra.StringAlgebra
###interpretation flavor: de.up.ling.irtg.algebra.StringAlgebra
###interpretation framework: de.up.ling.irtg.algebra.StringAlgebra
###interpretation version: de.up.ling.irtg.algebra.StringAlgebra
###interpretation time: de.up.ling.irtg.algebra.StringAlgebra
###interpretation spans: de.up.ling.irtg.algebra.StringAlgebra
###interpretation input: de.up.ling.irtg.algebra.StringAlgebra
###interpretation string: de.up.ling.irtg.algebra.StringAlgebra
###interpretation alignment: de.up.ling.irtg.algebra.StringAlgebra
###interpretation graph: de.up.ling.irtg.algebra.graph.GraphAlgebra
"""

data = []
errors = 0
total = 0
for filename in os.listdir(mrp_dir):
    if not filename.startswith('.'):
        print(filename)
        with open(mrp_dir + filename,encoding='utf8', errors='ignore') as infile:
            for line in infile:
                total += 1
                try:
                    mrp_dict = json.loads(line)
                    input = mrp_dict["input"]
                    id = mrp_dict["id"]
                    flavor = mrp_dict["flavor"]
                    framework = mrp_dict["framework"]
                    version = mrp_dict["version"]
                    time = mrp_dict["time"]
                    for token_file in os.listdir(tokenized_dir):
                        #print(token_file)
                        if token_file[:3] == filename[:3]:
                            companion_data = json.load(open(tokenized_dir+token_file, encoding='utf-8'))
                            if id not in companion_data.keys():
                                continue
                            else:
                                spans = ' '.join(list(companion_data[id]["spans"].keys()))
                                #spans = sorted(spans, key = lambda x:int(x.split(':')[0]))
                                tokens = companion_data[id]['tokenization']
                                edges = get_mrp_edges(mrp_dict)
                                edges = eliminate_h(edges)
                                labels = get_id2lex(mrp_dict)
                                compressed_edges = compress_c_edge(edges)
                                compressed_labels = update_id_labels(compressed_edges, labels)
                                irtg_format_compressed = edge2irtg(compressed_edges, labels)
                                node_tokens = node_to_token_index(companion_data, mrp_dict, compressed_labels, id)
                                aligned = percolate(compressed_edges, priority_queue, compressed_labels)
                                alignments = ''
                                for alignment in aligned.keys():
                                    for node in aligned[alignment]:
                                        if type(node) == str:
                                            if '<root>' in node:
                                                node = node[:-6]
                                        alignments += str(node) + '|'
                                    alignments += str(alignment)+'!' + '||' + str(node_tokens[alignment]) + '||' + '1.0 '
                                data.append((id, flavor, framework, version,time, spans, input, tokens, alignments, irtg_format_compressed))
                except:
                    errors += 1


train_mrp = []
dev_mrp = []
train_test_boundary = int((len(data)*80)/100)
random.Random(1).shuffle(data)
training = data[:train_test_boundary]
dev = data[train_test_boundary:]

print('percentage of training data skipped:')
print(errors/total)

with open(outdir+'training.txt', 'w') as outfile:
    outfile.write(header)
    for (id, flavor, framework, version,time, spans, input, tokens, alignments, irtg_format_compressed) in training:
        outfile.write(id)
        outfile.write('\n')
        outfile.write(str(flavor))
        outfile.write('\n')
        outfile.write(framework)
        outfile.write('\n')
        outfile.write(str(version))
        outfile.write('\n')
        outfile.write(time)
        outfile.write('\n')
        outfile.write(spans)
        outfile.write('\n')
        outfile.write(input)
        outfile.write('\n')
        outfile.write(' '.join(tokens))
        outfile.write('\n')
        outfile.write(alignments)
        outfile.write('\n')
        outfile.write(irtg_format_compressed)
        outfile.write('\n\n')

with open(outdir+'dev.txt', 'w') as outfile:
    outfile.write(header)
    for (id, flavor, framework,version,time, spans, input,tokens, alignments, irtg_format_compressed) in dev:
        outfile.write(id)
        outfile.write('\n')
        outfile.write(str(flavor))
        outfile.write('\n')
        outfile.write(framework)
        outfile.write('\n')
        outfile.write(str(version))
        outfile.write('\n')
        outfile.write(time)
        outfile.write('\n')
        outfile.write(spans)
        outfile.write('\n');
        outfile.write(input)
        outfile.write('\n')
        outfile.write(' '.join(tokens))
        outfile.write('\n')
        outfile.write(alignments)
        outfile.write('\n')
        outfile.write(irtg_format_compressed)
        outfile.write('\n\n')
print('written dev.txt')

train_ids = [i[0] for i in training]
training = None
dev_ids = [i[0] for i in dev]
dev = None

train_mrp = []
dev_mrp = []
for filename in os.listdir(mrp_dir):
    if not filename.startswith('.'):
        with open(mrp_dir + filename,encoding='utf8', errors='ignore') as infile:
            for id in train_ids:
                for line in infile:
                    mrp_dict = json.loads(line)
                    if id == mrp_dict['id']:
                        train_mrp.append(line)
print('appended train')

for filename in os.listdir(mrp_dir):
    if not filename.startswith('.'):
        with open(mrp_dir + filename,encoding='utf8', errors='ignore') as infile:
            for id in dev_ids:
                for line in infile:
                    mrp_dict = json.loads(line)
                    if id == mrp_dict['id']:
                        dev_mrp.append(line)
print('appended dev')
print('sanity check: number of examples in dev data')
print(len(dev_mrp))
with open(outdir + 'train.mrp', 'w') as outfile:
    for mrp in train_mrp:
        outfile.write(json.dumps(mrp))
        outfile.write('\n')
print('written train.mrp')
with open(outdir + 'dev.mrp', 'w') as outfile:
    for mrp in dev_mrp:
        outfile.write(json.dumps(mrp))
        outfile.write('\n')
print('written dev.mrp')
print('Done!')
