import sys
import json
import collections
import os
import random
from tqdm import tqdm

from edge_to_irtg import edge2irtg
from get_edges_from_mrp import get_id2lex, get_mrp_edges
from convert_irtg_to_mrp import get_edges, get_mrp_edges, get_nodes, get_tops, irtg2mrp
from eliminate_h_top import eliminate_h
from a_star_mrp import *
from process_c import *
from head_percolation_alignment import percolate
from move_edges import raise_edge
from get_mrp_from_intermediate import get_mrp
from utils import number_edges

mrp_dir = sys.argv[1]
tokenized_dir = sys.argv[2]
outdir = sys.argv[3]

priority_queue = 'L L-r L-l LR LA H H-l H-r P P-r P-l S S-r S-l A A-r A-l N C D T E R F G Q U'.split()
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


def reverse_dict(d):
    """
    reverses key-value mapping
    """
    r = dict()
    for k,v in d.items():
        r[v] = k
    return r

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
data_mrp = []
errors = 0
total = 0
for filename in os.listdir(mrp_dir):
    if not filename.startswith('.'):
        with open(mrp_dir + filename,encoding='utf8', errors='ignore') as infile:
            for line in tqdm(infile):
                total += 1
                try:
                    mrp_dict = json.loads(line)
                    input = mrp_dict["input"]
                    id = mrp_dict["id"]
                    #print(id)
                    flavor = mrp_dict["flavor"]
                    framework = mrp_dict["framework"]
                    version = mrp_dict["version"]
                    time = mrp_dict["time"]
                    for token_file in os.listdir(tokenized_dir):
                        if token_file[:3] == filename[:3]:
                            companion_data = json.load(open(tokenized_dir+token_file, encoding='utf-8'))
                            if id not in companion_data.keys():
                                continue
                            else:
                                span_dict = reverse_dict(companion_data[id]["spans"]) #keys tokens, values token ranges
                                spans = ' '.join([span_dict[position] for position in sorted(span_dict.keys())])
                                tokens = companion_data[id]['tokenization']
                                edges = get_mrp_edges(mrp_dict, get_remote =False)
                                edges = eliminate_h(edges)
                                labels = get_id2lex(mrp_dict)
                                compressed_edges = compress_c_edge(edges)
                                labels = update_id_labels(compressed_edges, labels)
                                raised_u = raise_edge(compressed_edges, 'U', ['L', 'H', 'P', 'S', 'D', 'T', 'Q', 'E', 'R', 'F', 'D'], label_dict=labels)
                                labels = update_id_labels(raised_u, labels)
                                raised_f = raise_edge(raised_u, 'F', ['L', 'H', 'P', 'S', 'D', 'T', 'Q', 'E', 'R', 'D'], label_dict=labels)
                                labels = update_id_labels(raised_f, labels)
                                raised_d = raise_edge(raised_f, 'D', ['L', 'H', 'P', 'S', 'T', 'Q', 'E', 'R', 'F'], label_dict=labels)
                                labels = update_id_labels(raised_d, labels)
                                raised_e = raise_edge(raised_d, 'E', ['L', 'H', 'P', 'S', 'T', 'Q', 'E', 'R', 'F', 'D'], label_dict=labels)
                                labels = update_id_labels(raised_d, labels)
                                raised_r = raise_edge(raised_e, 'R', ['L', 'H', 'P', 'S', 'T', 'Q', 'E', 'F', 'D'], label_dict=labels)
                                labels = update_id_labels(raised_d, labels)
                                raised_t = raise_edge(raised_r, 'T', ['L', 'H', 'P', 'S', 'Q', 'E', 'F', 'D'], label_dict=labels)
                                labels = update_id_labels(raised_t, labels)
                                raised_q = raise_edge(raised_t, 'Q', ['L', 'H', 'P', 'S', 'T', 'Q', 'E', 'F', 'D'], label_dict=labels)
                                updated_id_labels = update_id_labels(raised_q, labels)
                                irtg_format_raised = edge2irtg(raised_q, updated_id_labels)
                                node_tokens = node_to_token_index(companion_data, mrp_dict, updated_id_labels, id)
                                if id =='057386-0002':
                                    print(node_tokens)
                                #print(node_tokens)
                                aligned = percolate(raised_q, priority_queue, updated_id_labels)
                                alignments = ''
                                for alignment in aligned.keys():
                                    for node in aligned[alignment]:
                                        if type(node) == str:
                                            if '<root>' in node:
                                                node = node[:-6]
                                        alignments += str(node) + '|'
                                    alignments += str(alignment)+'!' + '||' + str(node_tokens[alignment]) + '||' + '1.0 '
                                if id =='057386-0002':
                                    print(alignments)
                                    print(aligned)
                                    print(node_tokens)
                                data.append((id, flavor, framework, version,time, spans, input, tokens, alignments, irtg_format_raised))
                                #print(raised)
                                new_mrp = get_mrp(id,flavor, framework, version, time, input, companion_data[id]['spans'], raised_r)
                                data_mrp.append(new_mrp)
                except:
                    errors += 1



#sys.exit()
train_test_boundary = int((len(data)*80)/100)
random.Random(1).shuffle(data)
training = data[:train_test_boundary]
test = data[train_test_boundary:]
#mrp_dev = data_mrp[train_test_boundary:]

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
    for (id, flavor, framework,version,time, spans, input,tokens, alignments, irtg_format_compressed) in test:
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

with open(outdir+'everything_raised.mrp', 'w') as out_mrp:
    for mrp in data_mrp:
        out_mrp.write(json.dumps(mrp))
        out_mrp.write('\n')
