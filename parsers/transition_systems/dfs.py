from copy import deepcopy
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch

from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.additional_lexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.transition_systems.parsing_state import ParsingState
from topdown_parser.transition_systems.transition_system import TransitionSystem
from .decision import Decision
from topdown_parser.transition_systems.unconstrained_system import UnconstrainedTransitionSystem


class DFSState(ParsingState):

    def is_complete(self) -> bool:
        return self.stack == []

    def copy(self) -> "ParsingState":
        return DFSState(self.decoder_state, self.active_node, self.score, self.sentence, self.lexicon,
                        list(self.heads), deepcopy(self.children), list(self.edge_labels), list(self.constants), list(self.lex_labels),
                        list(self.stack), set(self.seen))




class DFS(UnconstrainedTransitionSystem):

    def __init__(self, children_order: str, pop_with_0: bool, additional_lexicon : AdditionalLexicon):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        """
        super().__init__(additional_lexicon, pop_with_0)
        assert children_order in ["LR", "IO"], "unknown children order"

        self.children_order = children_order

    def _construct_seq(self, tree: Tree, is_first_child : bool, parent_type : Tuple[str, str], parent_lex_label : str) -> List[Decision]:
        own_position = tree.node[0]
        to_left = []
        to_right = []
        for child in tree.children:
            if child.node[1].label == "IGNORE":
                continue

            if child.node[0] < own_position:
                to_left.append(child)
            else:
                to_right.append(child)

        if is_first_child:
            beginning = [Decision(own_position, False, tree.node[1].label, parent_type, parent_lex_label)]
        else:
            beginning = [Decision(own_position, False, tree.node[1].label, ("", ""), "")]

        if self.children_order == "LR":
            children = to_left + to_right
        elif self.children_order == "IO":
            children = list(reversed(to_left)) + to_right
        else:
            raise ValueError("Unknown children order: " + self.children_order)

        ret = beginning
        for i, child in enumerate(children):
            ret.extend(self._construct_seq(child, i == 0, (tree.node[1].fragment, tree.node[1].typ), tree.node[1].lexlabel))

        last_position = 0 if self.pop_with_0 else own_position
        if len(tree.children) == 0:
            #This subtree has no children, thus also no first child at which we would determine the type of the parent
            #Let's determine the type now.
            last_decision = Decision(last_position, True, "", (tree.node[1].fragment, tree.node[1].typ),
                                     tree.node[1].lexlabel)
        else:
            last_decision = Decision(last_position, True, "", ("",""), "")
        ret.append(last_decision)
        return ret


    def get_order(self, sentence: AMSentence) -> Iterable[Decision]:
        t = Tree.from_am_sentence(sentence)
        r = self._construct_seq(t, False, ("",""),"")
        return r

    def get_unconstrained_version(self) -> TransitionSystem:
        """
        Return an unconstrained version that does not do type checking.
        :return:
        """
        return self

    def guarantees_well_typedness(self) -> bool:
        return False

    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        return all(x.head == y.head for x, y in zip(gold_sentence, predicted)) and \
               all(x.label == y.label for x, y in zip(gold_sentence, predicted)) and \
               all(x.fragment == y.fragment and x.typ == y.typ for x, y in zip(gold_sentence, predicted))

    def initial_state(self, sentence : AMSentence, decoder_state : Any) -> ParsingState:
        stack = [0]
        seen = set()
        heads = [0 for _ in range(len(sentence))]
        children = {i: [] for i in range(len(sentence) + 1)}
        labels = ["IGNORE" for _ in range(len(sentence))]
        lex_labels = ["_" for _ in range(len(sentence))]
        supertags = [("_","_") for _ in range(len(sentence))]

        return DFSState(decoder_state, 0, 0.0, sentence,self.additional_lexicon, heads, children, labels, supertags, lex_labels, stack, seen)

    def step(self, state : DFSState, decision: Decision, in_place: bool = False) -> ParsingState:
        if in_place:
            copy = state
        else:
            copy = state.copy()

        if state.stack:
            if decision.position == 0 and self.pop_with_0:
                copy.stack.pop()
            elif not self.pop_with_0 and decision.position in copy.seen:
                popped = copy.stack.pop()
                assert popped == decision.position
            else:
                copy.heads[decision.position-1] = copy.stack[-1]

                copy.children[copy.stack[-1]].append(decision.position)  # 1-based

                copy.edge_labels[decision.position - 1] = decision.label

                # push onto stack
                copy.stack.append(decision.position)

            if copy.constants is not None and copy.constants[state.active_node-1] == ("_","_") and state.active_node != 0:
                copy.constants[state.active_node-1] = decision.supertag

            if copy.lex_labels is not None and copy.lex_labels[state.active_node-1] == "_" and state.active_node != 0:
                copy.lex_labels[state.active_node-1] = decision.lexlabel

            copy.seen.add(decision.position)

            if not copy.stack:
                copy.active_node = 0
            else:
                copy.active_node = copy.stack[-1]
        else:
            copy.active_node = 0
        copy.score = copy.score + decision.score
        return copy
