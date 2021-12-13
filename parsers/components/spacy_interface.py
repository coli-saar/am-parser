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
This is used for predicting graphs from raw text and only intended for convenience,
it is NOT the pipeline used at test time in any of our experiments.
"""

from typing import List, Optional
import spacy
from spacy.tokens import Doc


def make_doc(vocab, sents : List[List[str]]) -> Doc:
    """
    Creates document with several sentences from nested list using a vocabulary.
    """
    d = Doc(vocab, [token for sentence in sents for token in sentence])
    i = 0
    for sent in sents:
        d[i].is_sent_start = True
        i += len(sent)
    return d


en_nlp = spacy.load("en_core_web_md", disable=["parser"])

def spacy_tokenize(sent : str) -> List[str]:
    words = en_nlp(sent)
    return [w.text.strip() for w in words if w.text.strip()]

lemma_dict = {"i": "I", "me": "I", "her": "she", "him": "he", "his": "he", "them": "they", "their": "they"}


def lemma_postprocess(token: str, lemma: str) -> str:
    if lemma == "-PRON-":
        if token.lower() in lemma_dict:
            return lemma_dict[token.lower()]
        else:
            return token
    return lemma


ne_dict = {"ORG": "ORGANIZATION", "LOC": "LOCATION", "NORP": "NATIONALITY", "QUANTITY": "NUMBER",
           "GPE": "COUNTRY"}  # GPE is NOT COUNTRY but we cannot split it into CITY, STATE_OR_PROVINCE easily so let's go with the majority


def is_number(tag: str) -> str:
    if tag in ["QUANTITY", "PERCENT", "CARDINAL", "MONEY"]:
        return "_number_"
    return "_name_"


def ne_postprocess(tag: str) -> str:
    if tag in ne_dict:
        return ne_dict[tag]
    if tag == "":
        return "O"
    return tag

def run_spacy(sents : List[List[str]], nlp = en_nlp) -> Doc:
    doc = make_doc(nlp.vocab, sents)
    for name,component in nlp.pipeline:
        component(doc)
    return doc
