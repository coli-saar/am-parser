#!/usr/bin/python3
#
# Copyright (c) 2020 Saarland University.
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
Compare two AMCONLL files

Prerequisites:
- Dot has to be installed (see https://graphviz.org/download/ )
- Inkscape has to be installed (see https://inkscape.org/ )
- command line `cat' and `tr' commands
- Pdflatex has to be installed (see https://www.latex-project.org/get/ )
- [except if --nogui] PyQt5 ( pip install pyqt5 ,
  see https://pypi.org/project/PyQt5/; for anaconda:  conda install -c anaconda pyqt)
- [for --viz] Imagemagick has to be installed
  (see https://imagemagick.org/index.php )

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

author: pia
tested using Ubuntu 18.04 , Python 3.7.4 , pyqt 5.9.2 , graphviz version 2.40.1,
Inkscape 0.92.3, pdfTeX 3.14159265-2.6-1.40.18 (TeX Live 2017/Debian),
ImageMagick 6.9.7-4
"""
# --viz --maxshow 10
# ../similarity2020/corpora/AMR/2017/gold-dev/gold-dev.amconll
# ../similarity2020/corpora/SemEval/2015/PSD/train/train.amconll

# todo: better comparison based on sentence as key (special chars...)
# todo: add set-random-number command line option
# todo: [enhancement] make is_interesting more robust wrt. sent length
# (art-root, nes?)
# todo: [enhancement] add fscore computation (is_interesting filter by fscore)
# todo: [enhancement] add visualization of graph itself, not its decomposition?
# todo: [enhancement][imagemagick] improve quality + scaling of images

import os
import subprocess
import argparse
import random
from tempfile import TemporaryDirectory
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
# needed for graph dependency parser imports:

from parsers.components.dataset_readers import amconll_tools
from parsers.components.dataset_readers.amconll_tools import AMSentence


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
        newtoken = token
        for old, new in repls:
            newtoken = newtoken.replace(old, new)
        if "-" in newtoken and " - " not in newtoken and newtoken != "-":
            # exclude if "-" token and tokes containing " - "
            # for all other hypens, add whitespace around them
            # right now also changes 10-10-2020 to 10 - 10 - 2020 ?
            newtokens.append(newtoken.replace("-", " - "))
        else:
            newtokens.append(newtoken)
    return newtokens


def is_interesting(instance: AMSentence) -> bool:
    """
    Define what to filter for.

    :param instance: AMSentence
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


def get_amsents(filename, use_id_as_key: bool = True) -> dict:
    """
    Read file, get id/sentencestring to AMSentence map

    :param filename: amconll file
    :param use_id_as_key: switch sentence id or sentence string equality
    :return: dictionry with id/sentencestring -> AMSentence
    """
    # todo [enhancement] input validation? file exists?
    graphs = dict()  # id -> AMSentence
    with open(file=filename, mode="r", encoding="utf-8") as fileobj:
        for sent in amconll_tools.parse_amconll(fileobj, validate=False):
            # todo what if sentence don't have id, but want to use id as keystr?
            # keystr = ''
            if not use_id_as_key:
                # sentence equality, instead of id equality
                # todo [enhancement] improve equality checks
                toks = sent.get_tokens(shadow_art_root=False)
                toks = normalize_toks(tokens=toks)
                keystr = ' '.join(toks)
                # if keystr.startswith("The total of"):
                #   print(keystr) # about double hypens
            else:
                keystr = sent.attributes["id"]
            graphs[keystr] = sent
    return graphs


def get_key_amsentpairs(file1: str, file2: str, use_id: bool = False) -> dict:
    """
    Read AMSentences from files, calculate intersection and return it

    :param file1: string, path to amconll file
    :param file2: string, path to amconll file
    :param use_id: whether an id should be used as key (and for equality check)
    :return: dict with id or sentence as key, and pair of AMSentence as value
    """
    assert(os.path.isfile(file1) and os.path.isfile(file2))
    am_sents_f1 = get_amsents(file1, use_id_as_key=use_id)
    am_sents_f2 = get_amsents(file2, use_id_as_key=use_id)
    common_keys = set.intersection(set(am_sents_f1.keys()),
                                   set(am_sents_f2.keys()))

    # print number of overlap sentences
    print(f";; Sentences in File 1:    {len(am_sents_f1)}")
    print(f";; Sentences in File 2:    {len(am_sents_f2)}")
    print(f";; Sentences in summed:    {len(am_sents_f2) + len(am_sents_f1)}")
    if len(am_sents_f1) + len(am_sents_f2) == 0:
        raise ValueError("No AM sentences found!")
    print(f";; Sentences in intersection: {len(common_keys)} ("
          f"{100*len(common_keys)/(len(am_sents_f1)+len(am_sents_f2)):3.2f} %)")
    # for sent in sorted(list(common_keys)):
    #     print(sent)

    if len(common_keys) == 0:
        # f_ks = sorted(am_sents_f1.keys())
        # g_ks = sorted(am_sents_f2.keys())
        # have you used id, but ids are not the same in both files?
        # do you compare the right files? (psd-train,dm-dev won't work)
        raise ValueError("No common sentences found!")

    # filter using is_interesting
    # todo KeyError possible if is_interesting returns True for only one file,
    #  but not for the other (e.g. function implemented such that it relies on
    #  framework specific things (art-root, tokenisation, edge name))
    am_sents_f1 = {k: v for k, v in am_sents_f1.items() if
                   is_interesting(v) and k in common_keys}
    am_sents_f2 = {k: v for k, v in am_sents_f2.items() if
                   is_interesting(v) and k in common_keys}
    key_to_f1f2amsent = {k1: (am_sents_f1[k1], v1)
                         for (k1, v1) in am_sents_f2.items()}
    # again with filter applied
    if len(am_sents_f1) == 0 or len(am_sents_f2) == 0:
        raise ValueError("No AM sentences found to compare! "
                         "Check your filter function")
    print(f";; Sentences after filtering: {len(key_to_f1f2amsent)} "
          f"({100 * (len(key_to_f1f2amsent)) / (len(common_keys)):3.2f} "
          f"% of common)")
    return key_to_f1f2amsent


def get_list_of_keys(d: dict, randomseed=None) -> list:
    keys = sorted(list(d.keys()))
    if randomseed is not None:
        # print(f";; Shuffle keys using random seed {str(randomseed)}")
        random.seed(randomseed)
        random.shuffle(keys)
    return keys


def main_nogui(sent_keys: list, am_f1gf: dict, use_id: bool,
               maxshow: int, visualize: bool):
    """
    Print the first maxshow elements and eventually visualizes them

    If visualize is True, uses imagemagick to visualize AMSentences
    :param sent_keys: list of keys of am_f1gf: ordering of sent. presentation
    :param am_f1gf: key is id/sentence, value if (AMSentence,AMSentence) pair
    :param use_id: whether the keys in sent_keys are ids or sentence strings
    :param visualize: if True, ImageMagick is used to visualize AMSentences
    :param maxshow:  print/show the first maxshow elements from target_key list
    :return: None
    """
    # in order of key list
    # - display 2 decompositions side by side and print to cmd id/sent
    with TemporaryDirectory() as direc:
        i = 0
        for key in sent_keys:
            # for each sentence (key either id or sent string)
            if i >= maxshow:
                break
            i += 1
            print(key)
            sent_f1, sent_gf = am_f1gf[key]
            if visualize:
                sentence = key
                if use_id:
                    sentence += " " + \
                                ' '.join(
                                    sent_gf.get_tokens(shadow_art_root=False))
                    print(sentence)
                    # print(' '.join(sent_f1.get_tokens(shadow_art_root=False)))
                sent_f1.to_tex_svg(direc, prefix="f1_")
                sent_gf.to_tex_svg(direc, prefix="gf_")

                # this relies on to_tex_svg creating files as sentence.tex and
                # pdflatex working
                if not os.path.isfile(os.path.join(direc, "f1_sentence2.svg")):
                    print(";; Warning: no svg output found- any error message?")
                    continue
                # todo [enhancement] label %f doesn't work   +"-label \'%f\'"
                # tile 1x2 (pics below each other, not next to each other)
                viz_cmd = "montage " + os.path.join(direc, "f1_sentence2.svg") \
                          + " " + os.path.join(direc, "gf_sentence2.svg") \
                          + f" -scale 150% -tile 1x2 -frame 5 " \
                            f"-title '{sentence}' -geometry +0+0 x:"
                # -resize 1512x1000
                with subprocess.Popen([viz_cmd], shell=True) as proc:
                    pass
    return


def main(argv):
    """
    Comparing two amconll files (at least, their intersection)

    Given two amconll files (system file and gold file), computes intersection
    and then
    - uses PyQt5 GUI to visualize common sentence decompositions **OR**
    - just prints sentences (--nogui) and eventually visualize them using
      Imagemagick (--viz)
    sentence ID equality (--useid) or sentence string equality
    (modulo some very basic handling for special characters and such) is used
    to find the intersection of both files.
    """
    optparser = argparse.ArgumentParser(
        add_help=True,
        description="compares two amconll files (can visualize)")
    optparser.add_argument("file1", help="system output", type=str)
    optparser.add_argument("gold_file", help="gold file", type=str)
    optparser.add_argument("--maxshow", type=int, default=1,
                           help="[noGUI only] maximum number of instances "
                                "to be shown (default %default)")
    optparser.add_argument("--useid", action="store_true",
                           help="use id instead of string equality")
    # todo [enhancement] random number argument (None or seed)
    optparser.add_argument("--viz", action="store_true",
                           help="[noGUI only] directly visualize")
    optparser.add_argument("--nogui", action="store_true",
                           help="don't use GUI")
    opts = optparser.parse_args(argv[1:])  # exclude script name

    file1 = opts.file1
    gold_file = opts.gold_file
    for file in [file1, gold_file]:
        if not os.path.isfile(file):
            raise RuntimeError(f"Not a valid file: {file}")

    # compute overlap
    use_id = opts.useid  # if False, uses sentence string, otherwise id
    am_f1gf = get_key_amsentpairs(use_id=use_id, file1=file1, file2=gold_file)
    # am_f1gf: id/sentence -> (AMSentence, AMSentence)

    # get list of keys of am_f1gf (optional: random shuffling)
    # remember keys are either sentence ids (--useid) or sentence strings (else)
    seed = 42
    if seed is not None:
        print(f";; Shuffle keys using random seed {str(seed)}")
    sent_keys = get_list_of_keys(d=am_f1gf, randomseed=seed)

    if opts.nogui:
        # just print sentences to stdout or use imagemagick
        main_nogui(sent_keys=sent_keys, am_f1gf=am_f1gf,
                   use_id=use_id, maxshow=opts.maxshow, visualize=opts.viz)
    else:
        from compare_amconll_qtgui import main_gui
        main_gui(sent_keys=sent_keys, am_f1gf=am_f1gf, use_id=use_id)
    return


if __name__ == '__main__':
    main(sys.argv)
