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


en_nlp = spacy.load("en_core_web_sm", disable=["parser"])

def run_spacy(sents : List[List[str]], nlp = en_nlp) -> Doc:
    doc = make_doc(nlp.vocab, sents)
    for name,component in nlp.pipeline:
        component(doc)
    return doc
