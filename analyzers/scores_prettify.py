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

Can print scores in a more human-readable format and display existence heatmaps

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

Can also display heatmaps of edge existence probabilities (`--showplot`):
see functions `heatmap_edge_existence_scores` and `which_sentence2plot`.
Currently specialized for COGS.

@author weissenh
tested with Python 3.7
"""
# python3 scores_prettify.py SCORESZIPFILE OUTPUTDIR  --prettify --showplot --maxshow 3
# python3 scores_prettify.py /PATH/TO/scores.zip /PATH/TO/OUTPUTDIR

# todo: test code
import io
import sys  # for argc,argv and exit
import os
import argparse  # command line argument parsing
import zipfile  # to read scores zip file
from io import TextIOWrapper  # utf-8 encoding for file in zip folder
from collections import defaultdict

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


################################################################################
# On plotting edge existence heatmaps ##########################################


# todo hard coding 'is interesting to plot'
# todo make more info available (scores, comments ...)?
def which_sentence2plot(tokens: list):
    """
    Should this sentence be plotted? (used for edge existence heatmap)

    :param tokens: list of strings, the tokens of the sentence
    :return: True if it should be plotted and False otherwise
    """
    preps = {"in", "on", "beside"}

    # if len(tokens) > 3 and tokens[2] in preps:  # obj_pp_to_subj_pp ?
    #     return True
    # count = sum(1 for token in tokens if token in {"that"})  # cp_recursion
    count = sum(1 for token in tokens if token in preps)  # pp_recursion
    if 5 <= count <= 12:  # if 3 <= count <= 5:
        return True
    return False


# colormaps reference: https://matplotlib.org/stable/gallery/color/colormap_reference.html
# "YlOrRd", "RdYlGn" "RdYlBu" "coolwarm" "viridis"
def heatmap_edge_existence_scores(from_to_ex: list, tokens: list, sentno: int,
                                  from_to_is_apply: list=None,
                                  nan_dummy: int = None,  # 0, -1
                                  cmap="RdYlBu"
                                  ):
    """
    Show heatmap of edge existence scores

    :param sentno: sentence number
    :param tokens: list of tokens including ART-ROOT as 0
    :param from_to_ex: list of lists  l[from][to] = score
    (score is logarithm of softmax or None)
    :param from_to_is_apply: matrix with True for APP, False for MOD, None else
    :param nan_dummy: how to plot NaN values? None means plot with same color
    as background. You can set it to a specific number if you would like it to
    show up on the color scale just like other 'normal' scores.
    :param cmap: color map for the heat map
    :
    >>> heatmap_edge_existence_scores(from_to_ex=[[None, -1.2, -0.1, -10], [None, None, -1.7, -8], [None, -0.2, None, -0.1], [None, -3, -1.7, None]], tokens=["ART-ROOT", "Ava", "slept", "tonight"], sentno=-1)
    """
    # from_to scores is a list of list
    nar = np.array(from_to_ex, dtype=np.float)
    nar = np.exp(nar)  # convert log (assume torch uses natural log) to exp

    # some values are NaN. do you want to make them appear on the color scale?
    if nan_dummy is not None and type(nan_dummy) == int:
        nar[np.isnan(nar)] = nan_dummy
    if from_to_is_apply is not None:
        # multiply all MODs with -1, all APPs with +1, IGNORE/ROOT are NaN
        is_apply = np.array(from_to_is_apply, dtype=np.bool)
        nar[is_apply == False] *= -1
        nar[is_apply is None] = np.nan  # todo check that this works
    vmin = -1 if from_to_is_apply is not None else nan_dummy  # todo used?

    # from_to_df = pd.DataFrame(nar, columns=tokens, index=tokens)
    # sns.heatmap(from_to_df, cmap="viridis")  # , annot=True
    # columns i = from token i || index j = to token j
    from_to_df = pd.DataFrame(nar, columns=tokens, index=tokens)
    # in principle we don't want any edge to end in art-root: del from_to_df["ART-ROOT"]

    assert from_to_df.max(skipna=True).max() <= 1.0  # softmax not beyond 1
    plt.imshow(from_to_df, cmap=cmap, vmin=vmin, vmax=1.0)
    plt.colorbar()

    def hide_some_tokens(ts: list) -> list:
        return ["" if x.lower() in {"a", "the"} else x for x in ts]

    # todo space between ticks such that they don't overlap
    plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.xticks(range(len(from_to_df)), hide_some_tokens(from_to_df.columns), rotation=80, fontsize=9)  # 60
    plt.yticks(range(len(from_to_df)), hide_some_tokens(from_to_df.index), fontsize=9)
    plt.xlabel("To")
    plt.ylabel("From")
    title = f"Heatmap for sentence {sentno}"
    if from_to_is_apply is not None:
        title += "\n>0 for APP, <0 for MOD, =0 ignore/root, white=NaN"
    plt.title(title)
    plt.grid(color="k", linewidth=.5)
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"../../test_{sentno}.png")
    #plt.close()
    return None


# todo add output dir to save hetamaps to?
def plot_heatmaps(scoreszipfile: str, maxshow=None):  # outputdir: str,
    print(f"Showing {maxshow} heatmaps...")  # todo neg value for max show to indicate: show all?
    shown_plots = 0
    for scores in get_next_sentence_scores(scoreszipfile=scoreszipfile):
        sentno, sent, ex_lb_scores, _ = scores
        existence, labels = ex_lb_scores
        tokens = sent.get_tokens(shadow_art_root=False)
        # todo can I already decide here whether to continue with next sent?
        should_plot = which_sentence2plot(tokens=tokens)
        if not should_plot:
            continue
        else:
            shown_plots += 1
        from_to_existence = [[None for __ in range(0, len(tokens) + 1)]
                             for _ in range(0, len(tokens) + 1)]
        from_to_is_apply = [[None for __ in range(0, len(tokens) + 1)]
                            for _ in range(0, len(tokens) + 1)]

        for fromidx, val in sorted(existence.items()):
            for toidx in sorted(val.keys()):
                existence_score = existence[fromidx][toidx]  # int or None
                labelscore_pairs = labels[fromidx][toidx]  # [(lbl, score), ...]
                assert 0 <= fromidx <= len(tokens)
                assert 0 <= toidx <= len(tokens)
                from_to_existence[fromidx][toidx] = existence_score

                # APP, MOD # todo make this distinction optional
                assert len(labelscore_pairs) > 0, "can't decide APP/MOD if no edge labels provided"
                label, lbl_score = labelscore_pairs[0]  # highest ranked
                if label.startswith("MOD"):
                    from_to_is_apply[fromidx][toidx] = False
                elif label.startswith("APP"):
                    from_to_is_apply[fromidx][toidx] = True
                elif label.startswith("ROOT") or label.startswith("IGNORE"):
                    from_to_is_apply[fromidx][toidx] = None
                    # todo is this for None really working?
                else:
                    print(label)
                    assert False  # shouldn't happen, assume exhaustive case distincition above
        heatmap_edge_existence_scores(
            from_to_ex=from_to_existence,
            from_to_is_apply=from_to_is_apply,
            tokens=["ART-ROOT"] + tokens,
            sentno=sentno
        )
        if shown_plots >= maxshow:
            # there might be more to plot, but not fo
            print(f"INFO: Finished plotting after {shown_plots} heatmaps")
            break
    return


################################################################################
# Pretty printing scores to output files #######################################


def pretty_print_edges(edges_outf: io.TextIOBase, tokens: list,
                       existence: dict, labels: dict,
                       edge_lbls_to_print: int=None):
    for fromidx, val in sorted(existence.items()):
        for toidx in sorted(val.keys()):
            existence_score = existence[fromidx][toidx]  # int or None
            labelscore_pairs = labels[fromidx][toidx]  # [(lbl, score), ...]

            assert 0 <= fromidx <= len(tokens)
            assert 0 <= toidx <= len(tokens)
            fromtoken = "ART-ROOT" if fromidx == 0 else tokens[fromidx - 1]
            totoken = "ART-ROOT" if toidx == 0 else tokens[toidx - 1]

            edges_outf.write(f"\tEdge {fromidx},{toidx} [{fromtoken}, {totoken}] with existence score {existence_score}\n")
            for lbl, lbl_score in labelscore_pairs[:edge_lbls_to_print]:
                edges_outf.write(f"\t\t{lbl_score}\t{lbl}\n")
    edges_outf.write("\n")
    return


def pretty_print_tags(tags_outf: io.TextIOBase, tokens: list,
                      supertag_scores: list, tags_to_print: int=None):
    # supertag_scores is a list, for each token sublist with (tag,score) pairs
    assert len(supertag_scores) == len(tokens)
    for i, tag_score_pairs in enumerate(supertag_scores):
        tags_outf.write(f"\tToken {i} [{tokens[i]}]:\n")
        for supertag, score in tag_score_pairs[:tags_to_print]:
            tags_outf.write(f"\t\t{score}\t{supertag}\n")
    tags_outf.write("\n")
    return


# todo add option to only print tags or edges
def pretty_print_scores(scoreszipfile: str, outputdir: str):
    print(f"Printing prettified scores for edges and tags to file")
    edge_lbls_to_print = 3  # todo make cmd param
    print(f"edge labels to print maximally: {edge_lbls_to_print}")
    tags_to_print = 3  # todo make cmd param
    assert tags_to_print > 0, "for 0 tags wouldn't make sense to print"
    print(f"tags to print maximally: {tags_to_print}")

    get_full_path = lambda x: os.path.join(outputdir, x)

    with open(get_full_path("tagProbs_prettified.txt"), "w", encoding="utf-8") as tags_outf, \
            open(get_full_path("opProbs_prettified.txt"), "w", encoding="utf-8") as edges_outf:
        # (sentno, AMsentence, ( existence[from][to]=s, labelscores[from][to]=(l,s) ), idx_supertag_score_pairs)
        for scores in get_next_sentence_scores(scoreszipfile=scoreszipfile):
            sentno, sent, ex_lb_scores, supertag_scores = scores
            tokens = sent.get_tokens(shadow_art_root=False)
            title = f"### Sentence Number {sentno}: {' '.join(tokens)}\n"

            # tags
            tags_outf.write(title)
            pretty_print_tags(tags_outf=tags_outf, tokens=tokens,
                              supertag_scores=supertag_scores,
                              tags_to_print=tags_to_print)

            # edges
            existence, labels = ex_lb_scores
            edges_outf.write(title)
            pretty_print_edges(edges_outf=edges_outf, tokens=tokens,
                               existence=existence, labels=labels,
                               edge_lbls_to_print=edge_lbls_to_print)
    return


################################################################################
# reading scores from the zip file #############################################


# ( existence[from][to] = int or None, labels[from][to] = [(label: str, score: int), ...] )
def get_edge_scores(line: str) -> tuple:
    # todo can't do input validation for indices range
    from_to_existence = defaultdict(lambda: defaultdict(None))  #existence[from][to] = None
    from_to_labels = defaultdict(lambda: defaultdict(list))  #labels[from][to] = []
    fromto_blocks = line.split(TAB_SPLIT)
    for i, fromto_block in enumerate(fromto_blocks):  # '[i,j]|existencescore edgelabel1|score1 edgelabel2|score2 ...'
        # for each possible edge
        blocks = fromto_block.split(WS_SPLIT)
        assert len(blocks) > 0
        fromto, score = tuple(blocks[0].split(PIPE_SPLIT))  # [i,j] , -3.14159
        fromidx, toidx = tuple(fromto[1:-1].split(","))
        fromidx, toidx = int(fromidx), int(toidx)
        assert 0 <= fromidx and 0 <= toidx
        from_to_existence[fromidx][toidx] = score

        for label_score_pair in blocks[1:]:  # edgelabel1|score1
            label, score = tuple(label_score_pair.split(PIPE_SPLIT))
            from_to_labels[fromidx][toidx].append((label, score))
    return from_to_existence, from_to_labels


# returns list, for each token a sublist with (tag,score) pairs
def get_supertag_scores(line: str) -> list:
    """
    One line is one sentence: top k (supertag, score) pairs for each token

    Format (according to the AM parser wiki):
    - for each token, there is a block of supertag scores.
    Blocks are separated by tabs.
    Each block contains multiple scores in a format like this:
    NULL|-0.0168 (d<root>__ALTO_WS__/__ALTO_WS__--LEX--)--TYPE--()|-4.69096
    , which says that the log probability for this token to have no contribution (NULL graph) is -0.0168 and
    that the log probability that this token has the supertag (d<root> / --LEX--)--TYPE--() is -4.691.
    Note that a space is used here to separate between the different supertags a token might have and thus,
    spaces in the graph constants are represented as __ALTO_WS__.
    :param line: one line of the tagProbs.txt
    :return: list, for each token idx a list of (supertag, socre) pairs
    """
    token_blocks = line.split(TAB_SPLIT)

    idx_supertag_score_pairs = [[] for _ in token_blocks]
    for i, token_block in enumerate(token_blocks):  # 'tag1|logscore1 tag2|logscore2 tag3|logscore3 ...'
        pairs = []
        # outfile.write(f"\tToken {i} [{tokens[i]}]:\n")
        supertag_score_pairs = token_block.split(WS_SPLIT)
        for supertag_score_pair in supertag_score_pairs:
            supertag, score = tuple(supertag_score_pair.split(PIPE_SPLIT))
            # todo is there a more performant version? for replace?
            supertag = supertag.replace(ALTOWS_SPLIT, " ")
            pairs.append((supertag, score))
            #outfile.write(f"\t\t{score}\t{supertag}\n")
        idx_supertag_score_pairs[i] = pairs
    return idx_supertag_score_pairs


# (sentno, AMsentence, ( existence[from][to]=s, labelscores[from][to]=(l,s) ), idx_supertag_score_pairs)
def get_next_sentence_scores(scoreszipfile: str) -> tuple:
    """
    Get AM Sentence, plus scores for edges and supertags: generator (yield)

    :param scoreszipfile: file path of the scores.zip file
    :return: generator, yields tuples of the following form
    (
      sentno: int,  # sentence number starting from 1
      sent: AMSentence,  # can do sent.get_tokens(shadow_art_root=False) to get list of str
      (existence, labelscores): pair # edge existence[fromid][toidx] = score, labelscores[from][to] = (label, score)
      idx_supertag_score_pairs: list  # for each token one sublist with (tag, score) pairs
    )
    All scores are log probabilities (natural log?)
    """
    with zipfile.ZipFile(scoreszipfile) as zf:
        with zf.open(SCORES_CORPUS_FILENAME, "r") as sentfile, \
                zf.open(SCORES_EDGES_FILENAME, "r") as edgefile, \
                zf.open(SCORES_TAGS_FILENAME, "r") as tagfile:
            sentno = 0
            for sent in amconll_tools.parse_amconll(TextIOWrapper(sentfile, 'utf-8'), validate=False):
                sentno += 1

                edge_line = edgefile.readline().decode("utf-8").strip()
                tag_line = tagfile.readline().decode("utf-8").strip()

                # toks = sent.get_tokens(shadow_art_root=False)  # List of str
                existence, labelscores = get_edge_scores(line=edge_line)
                idx_supertag_score_pairs = get_supertag_scores(line=tag_line)
                yield sentno, sent, (existence, labelscores), idx_supertag_score_pairs


################################################################################
# Main #########################################################################

def main(argv):
    """for usage call with --help option: python3 scores_prettify.py --help"""
    parser = argparse.ArgumentParser(
        add_help=True, description="Prettify and/or analyze scores.zip")
    parser.add_argument("scoreszipfile", type=str,
                        help="file path of the scores.zip file")
    parser.add_argument("outputdir", type=str,
                        help="output directory path: where to write output")
    # todo outputdir only needed when --prettify is given? or print heatmaps?

    parser.add_argument("--maxshow", dest="maxshow", type=int, default=10,
                        help="maximum number of heatmaps to be shown "
                             "(default: %(default)d)")
    parser.add_argument("--showplot", dest="showplot", action="store_true",
                        help="show edge existence heatmaps: up to maxshow ones")
    parser.add_argument("--prettify", dest="prettify", action="store_true",
                        help="write scores in prettified format to files")
    # todo options for maxshow edge labels, maxshow supertags
    # todo options for print only tags or edges prettified to file

    args = parser.parse_args(args=argv)
    maxshow, showplot, prettify = args.maxshow, args.showplot, args.prettify

    scoreszipfile = args.scoreszipfile
    outputdir = args.outputdir
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
                filenotfound = True
    if filenotfound:
        sys.exit(2)

    if showplot:
        plot_heatmaps(scoreszipfile=scoreszipfile, maxshow=maxshow)

    if prettify:
        pretty_print_scores(scoreszipfile=scoreszipfile, outputdir=outputdir)
    print("--Done!")
    return


if __name__ == "__main__":
    main(sys.argv[1:])
