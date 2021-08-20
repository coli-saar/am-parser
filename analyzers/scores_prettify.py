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
Prettify output of `../dump_scores.py`

Rationale: If you would like to analyze mistakes of the Astar parser and ask
questions like: did the model get supertag and/or edge right and just didn't
like their combination or was the right supertag/edge estimated to be unlikely
by the model?

On the format of the input score.zip file, see the coli wiki:
https://github.com/coli-saar/am-parser/wiki/Computing-scores#score-format :
A zip file with scores has the following contents:
- opProbs.txt with edge existence and edge label scores (op short for operation,
  since the edges represent AM algebra operations).
- tagProbs.txt with supertag scores
- corpus.amconll with the input sentences, the best unlabeled dependency tree
  according to the model, the best lexical labels and measurements of
  computation time (computation time of batch divided by batch size).

Can also display heatmaps of edge existence probabilities:
see functions `heatmap_edge_existence_scores` and `which_sentence2plot`.
Currently specialized for COGS

@author weissenh
tested with Python 3.7
"""
# python3 scores_prettify.py SCORESZIPFILE OUTPUTDIR
# python3 scores_prettify.py /PATH/TO/scores.zip /PATH/TO/OUTPUTDIR

# todo: test code

import sys  # for argc,argv and exit
import os
import zipfile  # to read scores zip file
from zipfile import ZipFile
from io import TextIOWrapper  # utf-8 encoding for file in zip folder

# playing around with heat maps for edge existence scores
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")  # Adds higher directory to python modules path.
# needed for graph dependency parser imports:
from graph_dependency_parser.components.dataset_readers import amconll_tools


TAB_SPLIT = "\t"
WS_SPLIT = " "
PIPE_SPLIT = "|"
ALTOWS_SPLIT = "__ALTO_WS__"


# these filenames are present in a scores.zip file produced by dump_scores.py
SCORES_EDGES_FILENAME = "opProbs.txt"
SCORES_TAGS_FILENAME = "tagProbs.txt"
SCORES_CORPUS_FILENAME = "corpus.amconll"

heatmaps_shown_count = 0


# todo hard coding 'is interesting to plot'
def which_sentence2plot(lineno: int, tokens: list):
    """
    Should this sentence be plotted? (used for edge existence heatmap)

    :param lineno: actually sentence number?
    :param tokens: list of strings, the tokens of the sentence
    :return: True if it should be plotted and False otherwise
    """
    global heatmaps_shown_count
    if 2 <= heatmaps_shown_count:
        return False
    # if lineno > 290:
    #     return False
    preps = {"in", "on", "beside"}

    count = 0
    for token in tokens:
        if token in preps:
            count += 1
    if 8 <= count <= 12:
        return True
    return False


def heatmap_edge_existence_scores(from_to_ex: list, tokens: list, sentno: int):
    """
    Show heatmap of edge existence scores

    :param from_to_ex: list of lists  l[from][to] = score
    (score is logarithm of softmax or None)
    :param tokens: list of tokens including ART-ROOT as 0
    :param sentno: sentence number
    >>> heatmap_edge_existence_scores(from_to_ex=[[None, -1.2, -0.1, -10], [None, None, -1.7, -8], [None, -0.2, None, -0.1], [None, -3, -1.7, None]], tokens=["ART-ROOT", "Ava", "slept", "tonight"], sentno=-1)
    """
    global heatmaps_shown_count
    heatmaps_shown_count += 1

    # from_to scores is a list of list
    nar = np.array(from_to_ex, dtype=np.float)
    nar = np.exp(nar)  # convert log (assume torch uses natural log) to exp

    # some values are NaN. We will use a dummy value here, so it can receive a
    # color. Chosen to be smaller than 0 (min value) to distinguish nan from 0
    # (remember log can be -inf, after exp it is 0.
    nan_dummy = -1
    nar[np.isnan(nar)] = nan_dummy

    # from_to_df = pd.DataFrame(nar, columns=tokens, index=tokens)
    # sns.heatmap(from_to_df, cmap="viridis")  # , annot=True
    # columns i = from token i || index j = to token j
    from_to_df = pd.DataFrame(nar, columns=tokens, index=tokens)
    # in principle we don't want any edge to end in art-root: del from_to_df["ART-ROOT"]

    cmap = "RdYlBu"  # "RdYlGn" "RdYlBu" "coolwarm" # "viridis"
    plt.imshow(from_to_df, cmap=cmap, vmin=nan_dummy, vmax=1.0)  # todo vmax softmax, so can't be beyond 1?
    plt.colorbar()

    def hide_some_tokens(ts: list) -> list:
        return ["" if x.lower() in {"a", "the"} else x for x in ts]

    plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.xticks(range(len(from_to_df)), hide_some_tokens(from_to_df.columns), rotation=60)  # -1
    plt.yticks(range(len(from_to_df)), hide_some_tokens(from_to_df.index))
    plt.xlabel("To")
    plt.ylabel("From")
    plt.title(f"Heatmap for sentence {sentno}")
    plt.grid(color="b", linewidth=.1)
    plt.tight_layout()
    plt.show()
    return None


def prettify_edges(zf: ZipFile, sents, outputfile):
    """
    Write inputfile content into outputfile in a more human-readable format

    It is your responsibility to ensure the sents argument matches input file sentence order
    :param zf  already opened scores.zip file object
    :param sents  list of sentences, each sentence is a list of tokens(strings)
    :param outputfile  path to output file (where results will be written to)
    In opProbs.txt,
    - each line corresponds to a sentence
    - the scores for different edges are tab-separated
    - for one edge, first there are the indices of the edge in the format [i,j] for an edge from i to j.
      The indices are 1-based with respect to the sentence, but there is a 0 used as an "artificial root",
      where ROOT and IGNORE edges attach. The score for the edge existence is given after a '|'; all scores are in log-space.
      After the edge existence score, the most likely labels and their scores are given, all separated by one whitespace.
      Example: the score
      [1,7]|-3.0764 MOD_mod|-0.7181 MOD_s|-0.9475
      means that the log probability of an edge existing from 1 to 7 is -3.07 and
      the log probability that this edge has the label MOD_mod is -0.7181,
      the log probability that this edge has the label MOD_s is -0.9475 and so on.
    """
    with zf.open(SCORES_EDGES_FILENAME) as infile, \
            open(outputfile, "w", encoding="utf-8") as outfile:
        lineno = 0
        for line in infile:
            # since this is opened from a zip archive, we need to decode
            # (the usual open(, encoding="utf-8") doesn't work with ZipFile )
            line = line.decode("utf-8")
            lineno += 1
            assert(lineno <= len(sents))
            tokens = sents[lineno-1]
            outfile.write(f"### Sentence Number {lineno}: {' '.join(tokens)}\n")

            # for heat map visualization:
            from_to_existence = [[None for __ in range(0, len(tokens)+1)] for _ in range(0, len(tokens)+1)]

            fromto_blocks = line.split(TAB_SPLIT)
            for i, fromto_block in enumerate(fromto_blocks): # '[i,j]|existencescore edgelabel1|score1 edgelabel2|score2 ...'
                # for each possible edge
                blocks = fromto_block.split(WS_SPLIT)
                assert(len(blocks) > 0)
                fromto, score = tuple(blocks[0].split(PIPE_SPLIT))  # [i,j] , -3.14159
                # get actual tokens: The indices are 1-based with respect to the sentence, but there is a 0 used as an "artificial root",
                fromidx, toidx = tuple(fromto[1:-1].split(","))
                fromidx, toidx = int(fromidx), int(toidx)
                assert(0 <= fromidx <= len(tokens) and 0 <= toidx <= len(tokens))
                fromtoken = "ART-ROOT" if fromidx == 0 else tokens[fromidx-1]
                totoken = "ART-ROOT" if toidx == 0 else tokens[toidx-1]

                # for heatmap visualization
                from_to_existence[fromidx][toidx] = score

                outfile.write(f"\tEdge {fromto} [{fromtoken}, {totoken}] with existence score {score}:\n")
                only_show_n_labels = 9  # debug set small value to reduce output
                for label_score_pair in blocks[1:only_show_n_labels+1]:
                    label, score = tuple(label_score_pair.split(PIPE_SPLIT))
                    outfile.write(f"\t\t{score}\t{label}\n")
            outfile.write("\n")

            # heatmap visualization
            if which_sentence2plot(lineno=lineno, tokens=tokens):
                heatmap_edge_existence_scores(from_to_ex=from_to_existence,
                                              tokens=["ART-ROOT"]+tokens,
                                              sentno=lineno)
    return


def prettify_tags(zf, sents, outputfile):
    """
    Write inputfile content into outputfile in a more human-readable format

    It is your responsiblity to ensure the sents argument matches input file sentence order
    :param zf  already opened scores.zip file object
    :param sents  list of sentences, each sentence is a list of tokens(strings)
    :param outputfile  path to output file (where results will be written to)
    In tagProbs.txt,
    - each line corresponds to a sentence
    - for each token, there is a block of supertag scores.
    Blocks are separated by tabs.
    Each block contains multiple scores in a format like this:
    NULL|-0.0168 (d<root>__ALTO_WS__/__ALTO_WS__--LEX--)--TYPE--()|-4.69096
    , which says that the log probability for this token to have no contribution (NULL graph) is -0.0168 and
    that the log probability that this token has the supertag (d<root> / --LEX--)--TYPE--() is -4.691.
    Note that a space is used here to separate between the different supertags a token might have and thus,
    spaces in the graph constants are represented as __ALTO_WS__.
    """
    with zf.open(SCORES_TAGS_FILENAME) as infile, \
            open(outputfile, "w", encoding="utf-8") as outfile:
        lineno = 0
        for line in infile:
            # since this is opened from a zip archive, we need to decode
            # (the usual open(, encoding="utf-8") doesn't work with ZipFile )
            line = line.decode("utf-8")
            lineno += 1
            assert (lineno <= len(sents))
            tokens = sents[lineno - 1]
            outfile.write(f"### Sentence Number {lineno}: {' '.join(tokens)}\n")
            token_blocks = line.split(TAB_SPLIT)
            assert(len(token_blocks)==len(tokens))

            for i, token_block in enumerate(token_blocks): # 'tag1|logscore1 tag2|logscore2 tag3|logscore3 ...'
                outfile.write(f"\tToken {i} [{tokens[i]}]:\n")
                supertag_score_pairs = token_block.split(WS_SPLIT)
                for supertag_score_pair in supertag_score_pairs:
                    supertag, score = tuple(supertag_score_pair.split(PIPE_SPLIT))
                    # todo is there a more performant version? for replace?
                    supertag = supertag.replace(ALTOWS_SPLIT, " ")
                    outfile.write(f"\t\t{score}\t{supertag}\n")
            outfile.write("\n")
    return


def main(argv):
    """for usage call with no arguments: python3 scores_prettify.py"""
    if len(argv) != 3:
        print("usage: python3 scores_prettify.py SCORESZIPFILE OUTPUTDIR")
        print("  -> prettify scores")
        print("  SCORESZIPFILE  file path of the scores zip file")
        print("  OUTPUTDIR      output directory path: where to write prettified score files")
        sys.exit(1)

    scoreszipfile = argv[1]
    outputdir = argv[2]
    # Check if file exists, outputdir exists
    if not os.path.isfile(scoreszipfile):
        print(f"ERROR: scores zip file doesn't exits. "
              f"Exit. File: {scoreszipfile} ", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(outputdir):
        print(f"ERROR: Output dir doesn't exist. "
              f"Exit. Path: {outputdir} ", file=sys.stderr)
        sys.exit(2)

    # check that zip file contains ...
    required_files = [SCORES_CORPUS_FILENAME,
                      SCORES_TAGS_FILENAME,
                      SCORES_EDGES_FILENAME]
    filenotfound = False
    with zipfile.ZipFile(scoreszipfile) as zf:
        files = zf.namelist()
        for required in required_files:
            if required not in files:
                print(f"ERROR: Score zip file doesn't contain the required "
                      f"file: {required} ", file=sys.stderr)
    if filenotfound:
        sys.exit(2)


    # Get sentences, each a list of tokens(strings):
    # We don't want to say 'Token 7' , but 'Token 7 [donkey]'
    with zipfile.ZipFile(scoreszipfile) as zf:
        sentences = list()
        with zf.open(SCORES_CORPUS_FILENAME, "r") as infile:  # , encoding="utf-8"
            for sent in amconll_tools.parse_amconll(TextIOWrapper(infile, 'utf-8'), validate=False):
                toks = sent.get_tokens(shadow_art_root=False)  # List of str
                sentences.append(toks)

        # Prettify tags (supertags) and edges (operations)
        get_full_path = lambda x: os.path.join(outputdir, x)
        prettify_tags(zf=zf, sents=sentences,
                      outputfile=get_full_path("tagProbs_prettified.txt"))
        prettify_edges(zf=zf, sents=sentences,
                       outputfile=get_full_path("opProbs_prettified.txt"))
    print("--Done!")
    return


if __name__ == "__main__":
    main(sys.argv)
