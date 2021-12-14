from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Set, Optional

import torch

from parsers.dataset_readers.amconll_tools import AMSentence
from parsers.dataset_readers.additional_lexicon import AdditionalLexicon

import numpy as np

class ParsingState(ABC):

    def __init__(self,decoder_state: Any, active_node: int, score: float,
                 sentence : AMSentence, lexicon : AdditionalLexicon,
                 heads: List[int], children: Dict[int, List[int]],
                 edge_labels : List[str], constants: List[Tuple[str,str]], lex_labels: List[str],
                 stack: List[int], seen: Set[int]):

        self.decoder_state = decoder_state
        self.active_node = active_node
        self.score = score
        self.lexicon = lexicon

        self.sentence = sentence
        self.heads = heads
        self.edge_labels = edge_labels
        self.constants = constants
        self.children = children
        self.lex_labels = lex_labels
        self.seen = seen
        self.stack = stack

    def copy(self) -> "ParsingState":
        """
        A way of copying this parsing state such that modifying objects that constrain the future
        will be modifying copied objects. e.g. we need a deep copy of the stack and nodes seen already
        but we don't need a deep copy of the decoder state or the lexicon.
        :return:
        """
        raise NotImplementedError()

    def extract_tree(self) -> AMSentence:
        sentence = self.sentence.set_heads(self.heads)
        sentence = sentence.set_labels(self.edge_labels)
        if self.constants is not None:
            sentence = sentence.set_supertag_tuples(self.constants)
        if self.lex_labels is not None:
            sentence = sentence.set_lexlabels(self.lex_labels)
        return sentence

    def is_complete(self) -> bool:
        raise NotImplementedError()

    def gather_context(self, device) -> Dict[str, torch.Tensor]:
        """
        Extracts features of the current context like children, parents etc.
        :param device: id of device which to put on the tensors.
        :return: a dictionary which can be used for storing
        various kinds of information, we can condition on.
        """

        # siblings: List[int] = get_siblings(self.children, self.heads, self.active_node)
        # labels_of_other_children = [self.edge_labels[child-1] for child in self.children[self.active_node]]

        # if self.constants is not None:
        #     if self.active_node == 0:
        #         supertag_of_current_node = "_"
        #     else:
        #         _, typ = self.constants[self.active_node-1]
        #         supertag_of_current_node = typ

        with torch.no_grad():
            ret = {"parents": torch.from_numpy(np.array([get_parent(self.heads, self.active_node)])).to(device)}
            # sibling_tensor = torch.zeros(max(1,len(siblings)), dtype=torch.long, device=device)
            # for j, sibling in enumerate(siblings):
            #     sibling_tensor[j] = sibling
            # ret["siblings"] = sibling_tensor
            # ret["siblings_mask"] = sibling_tensor != 0  # we initialized with 0 and 0 cannot possibly be a sibling of a node, because it's the artificial root.

            children_tensor = torch.zeros(max(1, len(self.children[self.active_node])), dtype=torch.long)
            for j, child in enumerate(self.children[self.active_node]):
                children_tensor[j] = child
            children_tensor = children_tensor.to(device)
            ret["children"] = children_tensor
            ret["children_mask"] = (children_tensor != 0) # 0 cannot be a child of a node.

            # if "edge_labels" in self.lexicon.sublexica:
            #     # edge labels of other children:
            #     label_tensor = torch.zeros(max(1, len(labels_of_other_children)), dtype=torch.long, device=device)
            #     for j, label in enumerate(labels_of_other_children):
            #         label_tensor[j] = self.lexicon.get_id("edge_labels", label)
            #     ret["children_labels"] = label_tensor
                #mask is children_mask

            # if "term_types" in self.lexicon.sublexica and self.constants is not None:
            #     ret["lexical_types"] = torch.tensor(np.array([self.lexicon.get_id("term_types", supertag_of_current_node)]), dtype=torch.long, device=device)

            return ret

def undo_one_batching(context : Dict[str, torch.Tensor]) -> None:
    """
    Undo the effects introduced by gathering context with batch size 1 and batching them up.
    This will mostly mean: do nothing or remove dimensions with size 1.
    :param context:
    """
    # context["parents"] has size (batch_size, decision seq len, 1)
    context["parents"] = context["parents"].squeeze(2)

    if "lexical_types" in context:
        context["lexical_types"] = context["lexical_types"].squeeze(2)


def undo_one_batching_eval(context : Dict[str, torch.Tensor]) -> None:
    """
    The same as above but at test time.
    :param context:
    :return:
    """
    # context["parents"] has size (batch_size, 1)
    context["parents"] = context["parents"].squeeze(1)

    if "lexical_types" in context:
        context["lexical_types"] = context["lexical_types"].squeeze(1)


def get_parent(heads : List[int], node : int) -> int:
    """
    Helper function to get the grandparent of a node.
    :param heads:
    :param node: 1-based
    :return: grandparent, 1-based
    """
    parent = heads[node-1]
    return parent


def get_siblings(children : Dict[int, List[int]], heads : List[int], node: int) -> List[int]:
    """
    Helper function to get siblings of a node.
    :param children: 1-based
    :param heads:
    :param node: 1-based
    :return: siblings, 1-based.
    """
    parent = heads[node-1]
    all_children_of_parent = list(children[parent])
    if node in all_children_of_parent:
        all_children_of_parent.remove(node)
    return all_children_of_parent







