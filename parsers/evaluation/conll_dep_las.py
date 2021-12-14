import argparse

from collections import namedtuple
import sys

### usage: python3 am_dep_las.py filename1 filename2
###
### Both files need to contain the same sentences in the same order + same format.
###
### author: ML

optparser = argparse.ArgumentParser(add_help=True,
    description="reads two conll files and computes UAS, LAS.")
optparser.add_argument("gold", type=str)
optparser.add_argument("system", type=str)

opts = optparser.parse_args()

head_col = 6
label_col = 7

Entry = namedtuple("Entry",["head", "label"])




def parse_amconll(fil):
    """
    Reads a file and returns a generator over AM sentences.
    :param fil:
    :return:
    """
    expect_header = True
    new_sentence = True
    entries = []
    attributes = dict()
    counter = 0
    for line in fil:
        line = line.rstrip("\n")
        if line.strip() == "":
            # sentence finished
            counter += 1
            if len(entries) > 0:
                if "id" not in attributes:
                    attributes["id"] = str(counter)
                yield entries, attributes
            new_sentence = True

        if new_sentence:
            expect_header = True
            attributes = dict()
            entries = []
            new_sentence = False
            if line.strip() == "":
                continue

        if expect_header:
            if line.startswith("#"):
                key, val = line[1:].split(":", maxsplit=1)
                attributes[key] = val
            else:
                expect_header = False

        if not expect_header:
            fields = line.split("\t")
            entries.append(Entry(fields[head_col], fields[label_col]))


tokens = 0
supertags_correct = 0
heads_correct = 0
labels_and_heads_correct = 0
lex_labels_correct = 0

with open(opts.gold) as f1:
    gold = { attr["id"] : s for s, attr in parse_amconll(f1) }

with open(opts.system) as f1:
    system = { attr["id"] : s for s, attr in parse_amconll(f1) }
    
print("Gold sents", len(gold))
print("System sents", len(system))
    
intersection = set(gold.keys()) & set(system.keys())

print("Intersection size", len(intersection))

for id in intersection:
    for gold_w, system_w in zip(gold[id], system[id]):
        tokens += 1
            
        if gold_w.head == system_w.head:
            heads_correct += 1
            if gold_w.label == system_w.label:
                labels_and_heads_correct += 1



if len(intersection) > 0:
    print()
    print("UAS (including IGNORE) %", round(heads_correct/tokens*100,3))
    print("LAS (including IGNORE) %", round(labels_and_heads_correct/tokens*100,3))

   

