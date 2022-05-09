from os import listdir
from os.path import isfile, join

import penman
import re
from collections import Counter

if __name__ == "__main__":
    dir_name = "../data/LDC2017T10/split/training/"
    onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    lemma2sense_counter = dict()
    for filename in onlyfiles:
        if filename.endswith(".txt"):
            print(dir_name+filename)
            with open(dir_name+filename, encoding="utf-8") as f:
                for graph in penman.load(f):
                    for node_triple in graph.instances():
                        label = node_triple[2]
                        match = re.search(r"-[0-9][0-9]$", label)
                        if match:
                            lemma = label[:-3]
                            sense = label[-2:]
                            # print(lemma+"-"+sense)
                            counter = lemma2sense_counter.get(lemma, Counter())
                            lemma2sense_counter[lemma] = counter
                            counter[sense] += 1

    sorted_keys = sorted(lemma2sense_counter.keys(), key=lambda l: len(lemma2sense_counter[l]), reverse=True)
    print(len(sorted_keys))
    with open("../example/visualization/senseDisambiguation/senseStatistics.txt", "w") as w:
        for key in sorted_keys:
            w.write(key+"\n")
            counter = lemma2sense_counter[key]
            for sense in sorted(counter.keys(), key=lambda s: counter[s], reverse=True):
                w.write(f"\t-{sense}: {counter[sense]}\n")
            w.write("\n")
