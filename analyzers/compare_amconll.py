#!/usr/bin/python3
"""
Compare two AMCONLL files

Prerequistites:
- Dot has to be installed (see https://graphviz.org/download/ )
- Ppdflatex has to be installed (see https://www.latex-project.org/get/ )
- Imagemagick (for visualization with --viz option) has to be installed (see https://imagemagick.org/index.php )

Setup:

1.) Set --maxhow, maximum number of sentences to be shown
    (if not specified, only 1 sentence is shown)
2.) Define is_interesting in way that suits you

Usage:

./compare_amconll.py example_compare_amconll/system_1.amconll example_compare_amconll/system_2.amconll

will first compute the overlap,
then filter according to the is_interesting function.
If there is still a non-empty set of common sentences,
will print these sentences in a random order,
but at most MAXSHOW sentences (or less if the set is smaller).
If --viz is specified,  will visualize the am dependency trees using imagemagick

author: pia (based on ML's compare_mrp.py)
"""
# --viz --maxshow 10
# ../similarity2020/corpora/AMR/2017/dev/dev.amconll
# ../similarity2020/corpora/SemEval/2015/PSD/train/train.amconll

# todo: better comparison based on sentence as key (special chars...)
# todo: switch from imagemagick to gui
# todo: quality and scaling of pictures should be improved
# todo: add set-random-number command line option
# todo: [enhancement] make is_interesting more robust wrt. sent length (art-root, nes?)
# todo: [enhancement] add fscore computation (filter by fscore)
# todo: [enhancement] add visualization of graph itself, not its decomposition?

import os
import subprocess
import argparse
import random
from tempfile import TemporaryDirectory
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
# needed for graph dependency parser imports:

from graph_dependency_parser.components.dataset_readers import amconll_tools
from graph_dependency_parser.components.dataset_readers.amconll_tools import AMSentence


def normalize_toks(tokens: list) -> list:
    # todo [enhancement] faster and smarter way to replace...
    # todo [enhancement] more replacements: html escaped chars, NEs
    repls = [("-LRB-", "("), ("-RRB-", ")"), ("’", "'"),
             ("_", " "), ("”", "''"), ("“", "``"), ("--", "–")]
    if tokens[-1] == "ART-ROOT":
        tokens = tokens[:-1]
        # toks[-1] = '.'
    newtokens = []
    for token in tokens:
        newt = token
        for old, new in repls:
            newt = newt.replace(old, new)
        if "-" in newt and " - " not in newt and newt != "-":
            # exclude if "-" token and tokes containing " - "
            # for all other hypens, add whitespace around them
            # right now also changes 10-10-2020 to 10 - 10 - 2020 ?
            newtokens.append(newt.replace("-", " - "))
        else:
            newtokens.append(newt)
    return newtokens


def is_interesting(instance: AMSentence):
    """
    Define what to filter for.
    instance: AMSentence
    """
    # todo [enhancement] possible to compute graph props like #edges, #nodes..?
    # todo [enhancement] fscore with other graph as criterium
    # what if different length due to one with ART-root different tokeniz?
    # better call normalize_toks, split whitespace and count
    tokens = normalize_toks(instance.get_tokens(shadow_art_root=False))
    newtokens = ' '.join(tokens).split(" ")
    if 5 < len(newtokens) < 15:
        return True
    # if 5 < len(instance) < 10:
    #     return True
    return False


def get_amsents(filename, use_id_as_key: bool = True):
    # todo [enhancement] input validation?
    graphs = dict()  # id -> AMSentence
    with open(file=filename, mode="r", encoding="utf-8") as fileobj:
        for _, sent in enumerate(amconll_tools.parse_amconll(fileobj, validate=False), 1):
            # todo what if sentence don't have id, but want to use id as keystr?
            # todo check sentence shadow art root really removed?
            # keystr = ''
            if not use_id_as_key:
                toks = sent.get_tokens(shadow_art_root=False)
                toks = normalize_toks(tokens=toks)
                keystr = ' '.join(toks)
                # if keystr.startswith("The total of"):
                #   print(keystr) # about double hypens
            else:
                # todo [enhancement] delete # from id? (for LaTeX or bash?)
                keystr = sent.attributes["id"]
            graphs[keystr] = sent
            # if opts.id == sent.attributes["id"] or (opts.id is None and i == opts.i):
            #    sent.to_tex_svg(opts.direc)
            #    found = True
            #    break
    return graphs

if __name__ == '__main__':
    optparser = argparse.ArgumentParser(
        add_help=True,
        description="compares two amconll files and spits out list of ids "
                    "with discrepencies/or visualizes them directly")
    optparser.add_argument("file1", help="system output", type=str)
    optparser.add_argument("gold_file", help="gold file", type=str)
    optparser.add_argument("--maxshow", type=int, default=1,
        help="maximum number of instances to be shown (default %default)")
    optparser.add_argument("--useid", action="store_true",
                           help="use id instead of string equality")
    # todo [enhancement] random number argument (None or seed)
    optparser.add_argument("--viz", action="store_true",
                           help="directly visualize")
    opts = optparser.parse_args()

    file1 = opts.file1
    gold_file = opts.gold_file
    for file in [file1, gold_file]:
        if not os.path.isfile(file):
            raise RuntimeError(f"Not a valid file: {file}")

    # compute overlap
    use_id = opts.useid  # if False, uses sentence string, otherwise id
    am_sents_f1 = get_amsents(file1, use_id_as_key=use_id)
    am_sents_gf = get_amsents(gold_file, use_id_as_key=use_id)
    common_keys = set.intersection(set(am_sents_f1.keys()), set(am_sents_gf.keys()))

    # print number of overlap sentences, (maybe ids?) (maybe also percentage?)
    print(f";; Sentences in File 1:       {len(am_sents_f1)}")
    print(f";; Sentences in Gold file:    {len(am_sents_gf)}")
    print(f";; Sentences in summed:       {len(am_sents_gf)+len(am_sents_f1)}")
    if len(am_sents_f1)+len(am_sents_gf) == 0:
        raise ValueError("No AM sentences found!")
    print(f";; Sentences in intersection: {len(common_keys)} ("
          f"{100*(len(common_keys)) / (len(am_sents_f1)+len(am_sents_gf)):3.2f} %)")
    # for sent in sorted(list(common_keys)):
    #     print(sent)
    if len(common_keys) == 0:
        g_ks = sorted(am_sents_gf.keys())
        f_ks = sorted(am_sents_f1.keys())
        raise ValueError("No common sentences found!")

    # filter
    # todo *if* is_interesting relies on different tokenizations +art-root (instance) -> KeyError
    am_sents_f1 = {k: v for k, v in am_sents_f1.items() if is_interesting(v) and k in common_keys}
    am_sents_gf = {k: v for k, v in am_sents_gf.items() if is_interesting(v) and k in common_keys}
    am_f1gf = {k1: (am_sents_f1[k1], v1) for (k1, v1) in am_sents_gf.items()}
    # again with filter applied
    if len(am_sents_f1) == 0 or len(am_sents_gf) == 0:
        raise ValueError("No AM sentences found to compare! Check your filter function")
    print(f";; Sentences after filtering: {len(am_f1gf)} "
          f"({100*(len(am_f1gf)) / (len(common_keys)):3.2f} % of common)")

    target_keys = list(am_f1gf.keys())
    target_keys = sorted(target_keys)
    random_number = 42
    if random_number:
        print(f";; Shuffle target_keys with random number {str(random_number)}")
        random.seed(random_number)
        random.shuffle(target_keys)
    # prompt: shuffle random number or None for no shuffling (can also make cmd arg)

    # in order of key list
    # - display 2 decompositions side by side (also graph?) and print to cmd id/sent
    # - enter to see close window
    with TemporaryDirectory() as direc:
        i = 0
        for key in target_keys:
            if i >= opts.maxshow:
                break
            i += 1
            print(key)
            sent_f1, sent_gf = am_f1gf[key]
            if opts.viz:
                sentence = key
                if use_id:
                    sentence += " " + \
                                ' '.join(sent_gf.get_tokens(shadow_art_root=False))
                    print(sentence)
                    # print(' '.join(sent_f1.get_tokens(shadow_art_root=False)))
                sent_f1.to_tex_svg(direc, prefix="f1_")  # am_sents_f1[key]
                # f2dot = os.path.join(direc, "f2.pdf")
                # with open(f2dot, "w") as f:
                #    f.write(sent_gf.to_tex_svg())
                sent_gf.to_tex_svg(direc, prefix="gf_")  # am_sents_gf[key]

                # this relies on to_tex_svg creating files as sentence.tex and pdflatex working
                if not os.path.isfile(os.path.join(direc, "f1_sentence2.svg")):
                    print(";; Warning: no svg output found - check error messages")
                    continue
                # todo [enhancement] label %f doesn't work   +"-label \'%f\'"
                # tile 1x2 (pics below each other, not next to each other: -tile x1 or default)
                viz_cmd = "montage "+os.path.join(direc, "f1_sentence2.svg")\
                          + " " +os.path.join(direc, "gf_sentence2.svg")\
                          + f" -scale 150% -tile 1x2 -frame 5 -title '{sentence}' -geometry +0+0 x:"
                # see also https://realpython.com/python-gui-tkinter/#check-your-understanding
                # or simpleGui? https://opensource.com/article/18/8/pysimplegui
                # note:  -resize 1512x1000 -scale 150% -adaptive-sharpen 150% -scale 300%
                with subprocess.Popen([viz_cmd], shell=True) as proc:
                    pass
