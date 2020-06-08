#!/usr/bin/python3
"""
Compare two AMCONLL files

Prerequistites:
- Dot has to be installed (see https://graphviz.org/download/ )
- Ppdflatex has to be installed (see https://www.latex-project.org/get/ )
- PyQt5 (pip install pyqt5 , see https://pypi.org/project/PyQt5/ )

Setup:

1.) Define is_interesting in way that suits you

Usage:

./compare_amconll_qtgui.py example_compare_amconll/system_1.amconll example_compare_amconll/system_2.amconll

will first compute the overlap,
then filter according to the is_interesting function.
If there is still a non-empty set of common sentences,
will shuffle remaining sentences and show them in a random order in a new window

author: pia
"""
# ../similarity2020/corpora/AMR/2017/gold-dev/gold-dev.amconll
# ../similarity2020/corpora/SemEval/2015/PSD/train/train.amconll

# tested on Ubuntu 18.04
# python 3.7.4
# pyqt 5.9.2
# pdfTeX 3.14159265-2.6-1.40.18 (TeX Live 2017/Debian)
# dot - graphviz version 2.40.1 (20161225.0304)

# todo: [GUI] get rid of blank space on top of main window
# todo: amr named entities with underscores cause problems?
#  (to-tex-svg escape needed?) also mod_UNIFY_s ?
# todo: better comparison based on sentence as key (special chars...)
# todo: add set-random-number command line option
# todo: [enhancement] make is_interesting more robust wrt. sent length
# todo: [enhancement] add fscore computation (filter by fscore)
# todo: [enhancement] add visualization of graph itself, not its decomposition?
# todo: [enhancement][GUI] list to scroll for sentences (~MaltEval)
# todo: [enhancement][GUI] save image (dialog window) to file.
# todo: [enhancement][GUI] border around images, keep ration during resize
# todo: [enhancement][GUI] select sentence by entering sentence number

import os
import argparse
import random
from tempfile import TemporaryDirectory
# GUI
from PyQt5.QtWidgets import QApplication, \
    QWidget, QLabel, QGridLayout, \
    QPushButton, QMainWindow, QDialog, QDialogButtonBox, QVBoxLayout
from PyQt5 import QtSvg, QtCore, Qt

import sys

from PyQt5 import QtGui

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
    with open(file=filename, mode="r") as fileobj:
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
            # if opts.id == sent.attributes["id"] or \
            #        (opts.id is None and i == opts.i):
            #    sent.to_tex_svg(opts.direc)
            #    found = True
            #    break
    return graphs


def get_key_amsentpairs(file1: str, file2: str, use_id: bool=False) -> dict:
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
    print(
        f";; Sentences in summed:       {len(am_sents_f2) + len(am_sents_f1)}")
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

    # filter
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


class DialogSvgMaximized(QDialog):
    """
    Dialog window class for enlarged svgs

    >>> app = QApplication([])
    >>> dlg = DialogSvgMaximized("filename.svg")
    >>> dlg.show()
    >>> sys.exit(app.exec_())
    """

    def __init__(self, svgfilename: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enlarged SVG image")
        # Set the central widget and the general layout
        self.dlgLayout = QVBoxLayout()
        self.setLayout(self.dlgLayout)
        # todo proper input validation
        assert(os.path.isfile(svgfilename))
        # todo [enhancement] maybe add buttons to save to file, zoom in/out???
        self.wdg_svg = QtSvg.QSvgWidget()
        self.wdg_svg.load(svgfilename)
        self.dlgLayout.addWidget(self.wdg_svg)
        # button to cancel
        self.btn_cnl = QPushButton("Close")
        self.btn_cnl.clicked.connect(self.close)
        # button to save file todo button save file
        # start here: https://pythonprogramming.net/file-saving-pyqt-tutorial/
        #self.btn_save = QPushButton("Save file")
        #self.btn_save.clicked.connect(self.sav)
        #self.dlgLayout.addWidget(self.btn_save)
        self.dlgLayout.addWidget(self.btn_cnl)
        self.showMaximized()  # full screen
        return


class PyCompareUi(QMainWindow):
    """PyCompare's View (GUI)."""

    def __init__(self, direc: TemporaryDirectory, useid: bool, amf1gf: dict,
                 target_keys: list):
        """
        Initializes GUI

        :param direc: TemporaryDirectory: where svg files and such are saved...
        :param useid: id or sentence as key? (-> displayed in sentence label)
        :param amf1gf: dict(key: str -> (file1: AMSentence, goldf: AMSentence))
        :param target_keys: keys for amf1gf (sorted)
        """
        super().__init__()
        # Set some main window's properties
        self.setWindowTitle("GUI PyQt5 Compare AMCoNLL files")
        # Set the central widget and the general layout
        self.generalLayout = QGridLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        # create and display some widgets
        self._create()
        # todo input validation (keys match amf1gf)
        if len(target_keys) == 0:
            raise RuntimeError
        self.useid = useid
        self.direc = direc  # .name
        self.amf1gf = amf1gf  # key -> (f1: Amconllsent,gf: Amconllsent)
        self.target_keys = target_keys
        self.total = len(target_keys)
        self.current_idx = 0
        self._update()
        self.showMaximized()  # full screen
        return

    def get_current_key(self):
        return self.target_keys[self.current_idx]

    def get_svgs(self):
        """
        Given the current key (and hence sentence) call to_tex_svg

        -> Need pdflatex and dot installed, assumes valid AMSentence and
        let's hope that there is no special character escaping stuff missing
        :raises RuntimeError if svg files couldn't be produced
        :return: pair of filepath to file1 and goldfile svgs
        """
        key = self.get_current_key()
        sent_f1, sent_gf = self.amf1gf[key]
        sent_f1.to_tex_svg(self.direc, prefix="f1_")  # am_sents_f1[key]
        sent_gf.to_tex_svg(self.direc, prefix="gf_")  # am_sents_gf[key]
        # this relies on to_tex_svg creating the necessary files
        fname_svg1 = os.path.join(self.direc, "f1_sentence2.svg")
        fname_svg2 = os.path.join(self.direc, "gf_sentence2.svg")
        if not os.path.isfile(fname_svg1) or not os.path.isfile(fname_svg2):
            # print(";; Warning: no svg output found - check error messages")
            # maybe pdflatex or dot had a problem (special characters?)
            raise RuntimeError("Couldn't find SVG files for sentence!")
        return fname_svg1, fname_svg2

    def get_sentence(self) -> str:
        """
        Get string representation of sentence (eventulally + id)

        Uses gold file string
        :return: sentence string. if self.useid, prefixed with id
        """
        _, sent_gf = self.amf1gf[self.get_current_key()]
        sentence = self.get_current_key()
        if self.useid:
            sentence += " " + \
                        ' '.join(sent_gf.get_tokens(shadow_art_root=False))
        return sentence

    def _update(self):
        """
        Call this function when a new sentence should be displayed

        Assuming the current_idx was already changed,
        updates the displayed information to reflect new sentence:
        - Changes sentence number
        - Changes sentence displayed
        - Changes the two svg images
        - Disables previous/next buttons if needed (new sentence is last/first)
        :return: None
        """
        # update displayed number
        self.lbl_no.setText(f"{self.current_idx+1} / {self.total}")
        # update displayed sentence
        sentence = self.get_sentence()
        self.lbl_sent.setText(sentence)
        # update images
        f1svg, gfsvg = self.get_svgs()
        self.svg1_filen = f1svg  # for enlarge dialog
        self.svg2_filen = gfsvg  # for enlarge dialog
        self.wdg_svg1.load(f1svg)
        self.wdg_svg2.load(gfsvg)
        # check if buttons need to be disabled (first/last sentence)
        self._eventually_disable_buttons()
        return

    def _eventually_disable_buttons(self):
        """Disables buttons if needed (for last and first sentence)"""
        isfirst = (self.current_idx == 0)
        self.btn_prev.setDisabled(isfirst)
        islast = (self.current_idx == len(self.target_keys)-1)
        self.btn_next.setDisabled(islast)
        return

    def _next_sent(self):
        """What needs to happen when the next sentence button is clicked"""
        assert(0 <= self.current_idx < len(self.target_keys))
        self.current_idx += 1
        self._update()
        return

    def _prev_sent(self):
        """What needs to happen when the previous sentence button is clicked"""
        assert(0 <= self.current_idx < len(self.target_keys))
        self.current_idx -= 1
        self._update()
        return

    def _enlarge_svg(self, filename:str):
        self.dlg = DialogSvgMaximized(filename)
        # block main window: need to close dialog in order to use main window
        self.dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        self.dlg.show()
        return

    def _enlarge_svg1(self):
        self._enlarge_svg(filename=self.svg1_filen)

    def _enlarge_svg2(self):
        self._enlarge_svg(filename=self.svg2_filen)

    def _create(self):
        """Create GUI"""
        # Sentence number (integer index in list, not identifier)
        height = 30
        font = QtGui.QFont('SansSerif', 13)
        self.lbl_no = QLabel(text="<No>", parent=self._centralWidget)
        self.lbl_no.setFont(font)
        self.lbl_no.setToolTip("Sentence no. X / Y total sentences")
        #self.lbl_no.setFixedSize(height, height)
        self.generalLayout.addWidget(self.lbl_no, 0, 0)
        # Sentence
        self.lbl_sent = QLabel(text="<Sentence>", parent=self._centralWidget)
        #self.lbl_sent.setFixedHeight(height)
        self.lbl_sent.setFont(font)
        self.lbl_sent.setToolTip("Current sentence")
        # setAlignment(Qt.AlignRight)
        # .setReadOnly(True)
        self.generalLayout.addWidget(self.lbl_sent, 0, 1)
        # buttons
        #  previous button
        self.btn_prev = QPushButton(text="Prev", parent=self._centralWidget)
        self.btn_prev.setFixedSize(height*2, height)
        self.btn_prev.setToolTip("Change to previous sentence")
        self.btn_prev.clicked.connect(self._prev_sent)
        self.generalLayout.addWidget(self.btn_prev, 0, 2)
        #  next button
        self.btn_next = QPushButton(text="Next", parent=self._centralWidget)
        self.btn_next.setFixedSize(height*2, height)
        self.btn_next.setToolTip("Change to next sentence")
        self.btn_next.clicked.connect(self._next_sent)
        self.generalLayout.addWidget(self.btn_next, 0, 3)
        # image 1
        # https://doc.qt.io/qt-5/qtsvg-index.html
        # https://stackoverflow.com/questions/44776474/display-svg-image-in-qtwebview-with-the-right-size
        self.wdg_svg1 = QtSvg.QSvgWidget(parent=self._centralWidget)
        # self.wdg_svg1.clicked.connect(self._enlarge_svg1)
        # self.wdg_svg1.load(file="")
        # row, col, rows spanned, cols spanned
        self.generalLayout.addWidget(self.wdg_svg1, 1, 0, 1, 3)
        self.wdg_svg2 = QtSvg.QSvgWidget(parent=self._centralWidget)
        # self.wdg_svg2.clicked.connect(self._enlarge_svg2)
        self.generalLayout.addWidget(self.wdg_svg2, 2, 0, 1, 3)
        # https://realpython.com/python-pyqt-gui-calculator/#learning-the-basics-of-pyqt
        # QDialog
        # todo [GUI][add] scroll list of sents
        # todo [GUI][add] add image maximize buttons, specify resizing...
        self.btn_enlarge1 = QPushButton(text="Max.", parent=self._centralWidget)
        self.btn_enlarge2 = QPushButton(text="Max.", parent=self._centralWidget)
        self.btn_enlarge1.setToolTip("Show image in separate window, maximized")
        self.btn_enlarge2.setToolTip("Show image in separate window, maximized")
        self.btn_enlarge1.clicked.connect(self._enlarge_svg1)
        self.btn_enlarge2.clicked.connect(self._enlarge_svg2)
        self.generalLayout.addWidget(self.btn_enlarge1, 1, 3)
        self.generalLayout.addWidget(self.btn_enlarge2, 2, 3)
        return


def main(argv):
    """Main function."""
    optparser = argparse.ArgumentParser(
        add_help=True,
        description="compares two amconll files and spits out list of ids "
                    "with discrepencies/or visualizes them directly")
    optparser.add_argument("file1", help="system output", type=str)
    optparser.add_argument("gold_file", help="gold file", type=str)
    optparser.add_argument("--useid", action="store_true",
                           help="use id instead of string equality")
    # todo [enhancement] random number argument (None or seed)
    opts = optparser.parse_args(argv[1:])

    file1 = opts.file1
    gold_file = opts.gold_file
    for file in [file1, gold_file]:
        if not os.path.isfile(file):
            raise RuntimeError(f"Not a valid file: {file}")

    # compute overlap
    use_id = opts.useid  # if False, uses sentence string, otherwise id
    am_f1gf = get_key_amsentpairs(use_id=use_id, file1=file1, file2=gold_file)

    target_keys = list(am_f1gf.keys())
    target_keys = sorted(target_keys)
    random_number = 42
    if random_number:
        print(f";; Shuffle target_keys with random number {str(random_number)}")
        random.seed(random_number)
        random.shuffle(target_keys)

    # Create an instance of QApplication
    app = QApplication([])
    # Show the x GUI
    with TemporaryDirectory() as direc:
        view = PyCompareUi(direc=direc, useid=use_id, amf1gf=am_f1gf, target_keys=target_keys)
        view.show()
        # Execute the calculator's main loop
        sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv)
