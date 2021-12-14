import argparse
import os

import sys
from typing import Dict, Set
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
from topdown_parser.dataset_readers.amconll_tools import parse_amconll

optparser = argparse.ArgumentParser(add_help=True,
                                    description="reads two amconll files annotated with (negative) log likelihoods and shows histogram of difference.")
optparser.add_argument("system",help="System prediction")
optparser.add_argument("gold",help="Log likelihood on gold data.")
optparser.add_argument("bins",type=int, default=200, help="Log likelihood on gold data.")
optparser.add_argument("--save",type=str, default=None, help="where to save figure.")

args = optparser.parse_args()

logl_system = []
logl_gold = []
lengths = []

with open(args.system) as system_f:
    with open(args.gold) as gold_f:
        system_trees = parse_amconll(system_f)
        gold_trees = parse_amconll(gold_f)
        for st, gt in zip(system_trees, gold_trees):
            lengths.append(len(st.words))
            logl_system.append(float(st.attributes["loss"]))
            logl_gold.append(float(gt.attributes["loss"]))

logl_sytem = np.array(logl_system)
logl_gold = np.array(logl_gold)
lenghts = np.array(lengths)

diff = logl_gold - logl_system

print("Plotting gold - system")
print("e.g. a gold tree with loss 10 and system tree with loss 15 appears as -5.")
print("That is, the more negative numbers there are, the more beam search is needed.")
print("Number of trees where search error could be found (difference is negative)", np.sum(diff < 0))

fig = plt.figure(figsize=(11,7))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
# spans two cols.
ax3 = fig.add_subplot(gs[1, :])

ax1.set_xlabel("Loss gold - system")
ax1.set_ylabel("Count")
ax1.hist(diff, bins=args.bins)

ax2.boxplot(diff)
ax2.set_ylabel("loss")
ax2.set_xlabel("gold - system")

ax3.scatter(lengths, diff,alpha=0.4)
ax3.set_xlabel("Sent length")
ax3.set_ylabel("gold - system")

if args.save:
    plt.savefig(args.save, dpi=300)
else:
    plt.show()