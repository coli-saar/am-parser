import argparse
import random

from graph_dependency_parser.components.dataset_readers.amconll_tools import parse_amconll, AMSentence, Entry, \
    write_conll

parser = argparse.ArgumentParser(description='Shuffle the alignments in an amconll file and mark it as unaligned.')

parser.add_argument('-i', '--input-file',
                    required=True,
                    type=str,
                    help='file path of the input amconll file')

parser.add_argument('-o', '--output-file',
                    required=True,
                    type=str,
                    help='file path to which the output amconll should be written')

args = parser.parse_args()

shuffled_sentences = []
for sentence in parse_amconll(open(args.input_file)):
    token2tagindex = [i for i in range(len(sentence))]
    random.shuffle(token2tagindex)
    lexlabels_shuffled = [sentence.get_lexlabels()[i] for i in token2tagindex]
    graphs_shuffled = [sentence.words[i].fragment for i in token2tagindex]
    types_shuffled = [sentence.words[i].typ for i in token2tagindex]
    heads_shuffled = []
    for i in token2tagindex:
        head = sentence.get_heads()[i]
        if head == 0:
            heads_shuffled.append(0)
        else:
            heads_shuffled.append(token2tagindex[head - 1] + 1)  # need to map both indices here, and take care of 0 vs 1
    edgelabels_shuffled = [sentence.get_edge_labels()[i] for i in token2tagindex]
    sentence.set_heads(heads_shuffled)
    sentence.set_lexlabels(lexlabels_shuffled)
    shuffled_sentences.append(AMSentence([Entry(w.token, w.replacement, w.lemma, w.pos_tag, w.ner_tag,
                                                graphs_shuffled[i], lexlabels_shuffled[i], types_shuffled[i],
                                                heads_shuffled[i], edgelabels_shuffled[i], False, w.range)
                                          for i, w in enumerate(sentence.words)], sentence.attributes))

write_conll(args.output_file, shuffled_sentences)
