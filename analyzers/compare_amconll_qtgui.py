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

Prerequistites:
- Dot has to be installed (see https://graphviz.org/download/ )
- Ppdflatex has to be installed (see https://www.latex-project.org/get/ )
- PyQt5 ( pip install pyqt5 , see https://pypi.org/project/PyQt5/ )

Setup:

1.) Define is_interesting in compare_amconll.py in way that suits you

Usage:

./compare_amconll_qtgui.py example_compare_amconll/system_1.amconll example_compare_amconll/system_2.amconll

will first compute the overlap,
then filter according to the is_interesting function.
If there is still a non-empty set of common sentences,
will shuffle remaining sentences and show them in a random order in a new window
Use the --useid option to compare sentence based on id rather than string
equality (e.g. AMR and PSD might not have the same ids, but DM and PSD have...)

author: pia
tested using Ubuntu 18.04 , Python 3.7.4 , pyqt 5.9.2 , graphviz version 2.40.1,
pdfTeX 3.14159265-2.6-1.40.18 (TeX Live 2017/Debian)
"""
# ../similarity2020/corpora/AMR/2017/gold-dev/gold-corpus.amconll
# ../similarity2020/corpora/SemEval/2015/PSD/train/train.amconll

# todo: add set-random-number command line option
# todo: [enhancement] add visualization of graph itself, not its decomposition?
# todo: [enhancement][GUI] list to scroll for sentences (~MaltEval)
# todo: [enhancement][GUI] save image (dialog window) to file.
# todo: [enhancement][GUI] border around images, keep ration during resize
# todo: [enhancement][GUI] select sentence by entering sentence number

import sys
import os
import argparse
from tempfile import TemporaryDirectory
# GUI
from PyQt5 import QtSvg, QtCore, QtGui
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout
from PyQt5.QtWidgets import QMainWindow, QDialog
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton

from compare_amconll import get_key_amsentpairs, get_list_of_keys


class DialogSvgMaximized(QDialog):
    """
    Dialog window class for enlarged SVGs

    Containing just this SVG (starts as maximized) and a 'Close' button
    >> app = QApplication([])
    >> dlg = DialogSvgMaximized("filename.svg")
    >> dlg.show()
    >> sys.exit(app.exec_())
    """

    def __init__(self, svgfilename: str, parent=None):
        """
        Initialize maximized dialog window displaying SVG (and a 'Close' button)

        :param svgfilename: File to be displayed inside the dialog window
        """
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
        # button to save file todo button save file (or put in PyCompareUi)
        # start here: https://pythonprogramming.net/file-saving-pyqt-tutorial/
        # self.btn_save = QPushButton("Save file")
        # self.btn_save.clicked.connect(self.SAVEFUN-not-implemented-yet)
        # self.dlgLayout.addWidget(self.btn_save)
        self.dlgLayout.addWidget(self.btn_cnl)
        self.showMaximized()  # full screen
        return


class PyCompareUi(QMainWindow):
    """PyCompare's View (GUI)."""

    def __init__(self, direc: TemporaryDirectory, useid: bool, amf1gf: dict,
                 sent_keys: list):
        """
        Initializes GUI

        :param direc: TemporaryDirectory: where svg files and such are saved...
        :param useid: id or sentence as key? (-> displayed in sentence label)
        :param amf1gf: dict(key: str -> (file1: AMSentence, goldf: AMSentence))
        :param sent_keys: keys for amf1gf (sorted)
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
        if len(sent_keys) == 0:
            raise RuntimeError
        self.useid = useid
        self.direc = direc  # .name
        self.amf1gf = amf1gf  # key -> (f1: Amconllsent,gf: Amconllsent)
        self.sent_keys = sent_keys
        self.total = len(sent_keys)
        self.current_idx = 0
        self._update()
        self.showMaximized()  # full screen
        return

    def get_current_key(self):
        return self.sent_keys[self.current_idx]

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
        islast = (self.current_idx == len(self.sent_keys) - 1)
        self.btn_next.setDisabled(islast)
        return

    def _next_sent(self):
        """What needs to happen when the next sentence button is clicked"""
        assert(0 <= self.current_idx < len(self.sent_keys))
        self.current_idx += 1
        self._update()
        return

    def _prev_sent(self):
        """What needs to happen when the previous sentence button is clicked"""
        assert(0 <= self.current_idx < len(self.sent_keys))
        self.current_idx -= 1
        self._update()
        return

    def _enlarge_svg(self, filename: str):
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
        # font = QtGui.QFont('SansSerif', 13)  # default font is quite tiny
        self.lbl_no = QLabel(text="<No>", parent=self._centralWidget)
        # self.lbl_no.setFont(font)
        self.lbl_no.setToolTip("Sentence no. X / Y total sentences")
        # self.lbl_no.setFixedSize(height, height)
        self.generalLayout.addWidget(self.lbl_no, 0, 0)
        # Sentence
        self.lbl_sent = QLabel(text="<Sentence>", parent=self._centralWidget)
        # self.lbl_sent.setFixedHeight(height)
        # self.lbl_sent.setFont(font)
        self.lbl_sent.setToolTip("Current sentence")
        # setAlignment(Qt.AlignRight)
        # .setReadOnly(True)
        self.generalLayout.addWidget(self.lbl_sent, 0, 1)
        # buttons
        #  'previous' button
        self.btn_prev = QPushButton(text="Prev", parent=self._centralWidget)
        self.btn_prev.setFixedSize(height*2, height)
        self.btn_prev.setToolTip("Change to previous sentence")
        self.btn_prev.clicked.connect(self._prev_sent)
        self.generalLayout.addWidget(self.btn_prev, 0, 2)
        #  'next' button
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
        # 'Maximize' buttons
        # todo [GUI][add] scroll list of sents
        # todo [GUI][add] image save2file, resizing/scaling and minSize?, ...
        self.btn_enlarge1 = QPushButton(text="Max.", parent=self._centralWidget)
        self.btn_enlarge2 = QPushButton(text="Max.", parent=self._centralWidget)
        self.btn_enlarge1.setToolTip("Show image in separate window, maximized")
        self.btn_enlarge2.setToolTip("Show image in separate window, maximized")
        self.btn_enlarge1.clicked.connect(self._enlarge_svg1)
        self.btn_enlarge2.clicked.connect(self._enlarge_svg2)
        self.generalLayout.addWidget(self.btn_enlarge1, 1, 3)
        self.generalLayout.addWidget(self.btn_enlarge2, 2, 3)
        return


def main_gui(sent_keys: list, am_f1gf: dict, use_id: bool):
    """
    Starts PyQt5 GUI comparing to amconll files (their intersection)

    :param sent_keys: list of keys of am_f1gf: ordering of sent. presentation
    :param am_f1gf: key is id/sentence, value if (AMSentence,AMSentence) pair
    :param use_id: whether the keys in sent_keys are ids or sentence strings
    :return: None
    """
    # Note: no input validation is done: specifically if all k in sent_keys
    # are valid keys of am_f1gf
    app = QApplication([])
    with TemporaryDirectory() as direc:  # for svg, tex files..
        view = PyCompareUi(direc=direc, useid=use_id, amf1gf=am_f1gf,
                           sent_keys=sent_keys)
        view.show()
        # exec_ listens for events
        sys.exit(app.exec_())


def main(argv):
    """
    Start PyQt5 GUI comparing two amconll files (at least, their intersection)

    Given two amconll files (system file and gold file), computes intersection
    and displays it in a GUI. Sentence equality is either determined by
    sentence ID equality (--useid) or sentence string equality
    (modulo some very basic handling for special characters and such).
    """
    optparser = argparse.ArgumentParser(
        add_help=True,
        description="compares two amconll files (GUI version)")
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

    # get list of keys of am_f1gf (optional: random shuffling)
    # remember keys are either sentence ids (--useid) or sentence strings (else)
    seed = 42
    if seed is not None:
        print(f";; Shuffle keys using random seed {str(seed)}")
    target_keys = get_list_of_keys(d=am_f1gf, randomseed=seed)

    # start GUI
    main_gui(sent_keys=target_keys, am_f1gf=am_f1gf, use_id=use_id)
    return


if __name__ == '__main__':
    main(sys.argv)
