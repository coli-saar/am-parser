"""
Takes an amconll files and reads off the sentences/literals/pos in the convention of sentences.txt of the AMR pipeline.

Call this file from the scripts directory with python extract_sentences.py path/to/amconll output/path/
"""

import sys
from os.path import join
sys.path.append("..")

from graph_dependency_parser.components.dataset_readers.amconll_tools import parse_amconll

sentences = []
literals = []
pos = []

with open(sys.argv[1]) as f:
    sents = parse_amconll(f)
    for sent in sents:
        tokens = sent.get_tokens(False)
        replacements = sent.get_replacements()
        o = []
        for t, r in zip(tokens, replacements):
            if r != "_":
                o.append(r.lower())
            else:
                o.append(t.lower())
        sentences.append(" ".join(o))
        literals.append(" ".join(tokens))
        pos.append(" ".join(sent.get_pos()))

with open(join(sys.argv[2],"sentences.txt"),"w") as f:
    for s in sentences:
        f.write(s)
        f.write("\n")

with open(join(sys.argv[2],"literal.txt"),"w") as f:
    for s in literals:
        f.write(s)
        f.write("\n")

with open(join(sys.argv[2],"pos.txt"),"w") as f:
    for s in pos:
        f.write(s)
        f.write("\n")

with open(join(sys.argv[2],"goldAMR.txt"),"w") as f:
    for s in pos:
        f.write("(n / no-gold-AMR)\n")


