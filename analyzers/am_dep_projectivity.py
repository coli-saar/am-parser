#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Saarland University.
#
# This file is part of AM Parser
# (see https://github.com/coli-saar/am-parser/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Compute block degrees (projectivity) for an amconll file

Rationale: the fixed-tree decoder (unlike the projective/A* one) can produce
non-projective AM dependency trees.

@author weissenh
tested with Python 3.7

similar project:
https://github.com/coli-saar/am-tools/blob/cogs_new_decomp/src/main/java/de/saar/coli/amtools/decomposition/analysis/ProjectivityVisualization.java
"""

# todo: test code (is it correct?)

import sys  # for argc,argv and exit
import os
from collections import Counter
from copy import deepcopy

sys.path.append("..")  # Adds higher directory to python modules path.
# needed for graph dependency parser imports:
from graph_dependency_parser.components.dataset_readers import amconll_tools
from graph_dependency_parser.components.dataset_readers.amconll_tools import AMSentence


# can I use memoization? (lru_cache won't work with dicts: unhashable type)
def transitive_closure_children(node: int, parent_of: dict, children_of: dict) -> set:
    # returns all children of node (not only direct children, but also grandchildren, grandgrandchildren...)
    children = children_of[node]
    if len(children) == 0:  # base case: if no direct children, set of all children is empty too
        return set()
    else:  # has children
        all_children = set()
        for child in children:
            all_children.add(child)  # direct child of node
            nodes = transitive_closure_children(child, parent_of=parent_of, children_of=children_of)
            all_children.update(nodes)  # indirect children of node
        return all_children


def determine_block_degree(node: int, children: set, ignored_ids: set) -> int:
    """
    For a given node, compute its block degree

    :param node: node for which to calculate block degree (ID, 1-based)
    :param children: set of dominated nodes (direct or indirect), set of IDs
    :param ignored_ids: ignored nodes
    :return: block degree (remember 1 means projective, >1 non-projective)
    """
    # todo what about root?
    if len(children - ignored_ids) <= 1:
        return 1  # if max. 1 child, block degree is trivial
    dominated_nodes = {node}
    dominated_nodes.update(children)
    imin = min(dominated_nodes)
    imax = max(dominated_nodes)

    degree = 1
    at_boundary = False
    blocks = list()
    blocks.append([imin])
    for nid in range(imin, imax+1):
        assert (len(blocks) == degree)
        is_inside = nid in dominated_nodes
        is_ignored = nid in ignored_ids
        if is_ignored and is_inside:
            # first asserted this, but if dep-tree isn't well-formed this can happen
            # (e.g. ignore edge head isn't 0, but some other id
            raise RuntimeError(f"Node {nid} can't be ignored and a dominated at the same time")

        if is_inside:  # at boundary or not, doesn't matter
            at_boundary = False
            blocks[degree - 1].append(nid)
        elif is_ignored and not at_boundary:
            # at_boundary = False  # that's already the case
            blocks[degree - 1].append(nid)
        elif is_ignored and at_boundary:
            continue  # we remain at boundary and don't include it in any block
        elif not is_inside and not is_ignored and at_boundary:
            continue  # we remain at boundary and don't include it in any block
        elif not is_inside and not is_ignored and not at_boundary:
            # new boundary found
            at_boundary = True
            degree += 1
            blocks.append([])
        else:
            assert(False)  # this case shouldn't occur
    assert (len(blocks) == degree)
    return degree


def analyze_projectivity(sent: AMSentence) -> int:
    """
    Analyses projectivity for a given AM dependency tree

    :param sent: AMSentence
    :returns block degree of the tree (1 if projective, higher otherwise)
    """
    # step 1: get parent-directchildren map for every word  O(sentlen)
    id2entry = {id+1: entry for id, entry in enumerate(sent.words)}  # id 1-based

    ignored_ids = {id for id, entry in id2entry.items() if entry.label == "IGNORE"}
    root_ids = {id for id, entry in id2entry.items() if entry.label == "ROOT"}
    #assert (len(root_ids) == 1)  # assume there is a unique rootfor now
    # if len(root_ids) != 1:
    #     print(f"For sentence length {len(sent)} and words: {' '.join(sent.get_tokens(shadow_art_root=False))}: not 1 uniqe root but {len(root_ids)} roots")
    if not(len(root_ids) <= 1):
        assert(False)
    #root_id = root_ids.pop()

    # parent of, child of dicts
    parent_of = {id: entry.head for id, entry in id2entry.items()}
    # parent_of[0] = None  # 0 indicates special node, doesn't have a parent
    children_of = {id: set() for id, entry in id2entry.items()}
    children_of[0] = set()  # 0 isn't in parent_of, because it isn't in id2entry
    for child, parent in parent_of.items():
        children_of[parent].add(child)

    # for each node, collect all nodes it dominates (even indirectly)
    all_children_of = deepcopy(children_of)  # so children_of stays the same todo needed?
    for id, entry in id2entry.items():
        all_children_of[id] = transitive_closure_children(node=id, parent_of=parent_of, children_of=children_of)
    # block degrees:
    block_degrees = dict()
    for id, entry in id2entry.items():
        degree = determine_block_degree(id, all_children_of[id], ignored_ids)
        block_degrees[id] = degree
    tree_block_degree = max(v for v in block_degrees.values())
    return tree_block_degree


def main(argv):
    """for usage call with no arguments: python3 am_dep_projectivity.py"""
    if len(argv) != 2:
        print("usage: python3 am_dep_projectivity.py AMCONLLFILE")
        print("  -> projectivity counts")
        print("  AMCONLLFILE path to an amconll file with dep trees")
        sys.exit(1)
    amconllfile = argv[1]
    # Check if file exists
    if not os.path.isfile(amconllfile):
        print(f"ERROR: File doesn't exits. Exit.  File: {amconllfile} ", file=sys.stderr)
        sys.exit(2)

    bd_counter = Counter()
    sentence_count = 0
    with open(amconllfile, "r", encoding="utf-8") as infile:
        for sent in amconll_tools.parse_amconll(infile, validate=False):
            sentence_count += 1
            try:
                dg = analyze_projectivity(sent=sent)
                bd_counter[dg] += 1
            except RuntimeError as err:
                print(f"Error at sentence no {sentence_count}: {err}")
                bd_counter["error"] += 1

    projective = bd_counter[1]
    all_count = sum(bd_counter.values())  # assert equals sentence count?
    print(f"#Sentences seen: {all_count:>10}")
    perc = 0.0 if all_count == 0 else (projective / float(all_count))*100
    print(f"#projective:     {projective:>10}  (that's {perc:6.2f}%)")
    print(f"Per block degree: (degree : count)")
    for degree, count in bd_counter.most_common():
        print(f"\t{degree:<3} : {count:>10}")

    print("--Done!")
    return


if __name__ == "__main__":
    main(sys.argv)
