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
#from topdown_parser.transition_systems.parsing_state import get_parent, get_siblings
from topdown_parser.transition_systems.unconstrained_system import UnconstrainedTransitionSystem


class DFSChildrenFirstState(ParsingState):

    def __init__(self, decoder_state: Any, active_node: int, score: float,
                 sentence : AMSentence, lexicon : AdditionalLexicon,
                 heads: List[int], children: Dict[int, List[int]], edge_labels: List[str],
                 constants : List[Tuple[str,str]], lex_labels : List[str],
                 stack : List[int], seen : Set[int], substack : List[int]):

        super().__init__(decoder_state, active_node, score, sentence, lexicon, heads, children, edge_labels,
                         constants, lex_labels, stack, seen)

        self.substack = substack
        self.step = 0

    def is_complete(self) -> bool:
        return self.stack == []

    def copy(self) -> "ParsingState":
        copy = DFSChildrenFirstState(self.decoder_state, self.active_node, self.score, self.sentence, self.lexicon,
                            list(self.heads), deepcopy(self.children), list(self.edge_labels), list(self.constants), list(self.lex_labels),
                            list(self.stack), set(self.seen), list(self.substack))
        copy.step = self.step
        return copy


class DFSChildrenFirst(UnconstrainedTransitionSystem):
    """
    DFS where when a node is visited for the second time, all its children are visited once.
    Afterwards, the first child is visited for the second time. Then the second child etc.
    """

    def __init__(self, children_order: str, pop_with_0: bool,
                 additional_lexicon: AdditionalLexicon,
                 reverse_push_actions: bool = False):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        reverse_push_actions means that the order of push actions is the opposite order in which the children of
        the node are recursively visited.
        """
        super().__init__(additional_lexicon, pop_with_0)
        self.reverse_push_actions = reverse_push_actions
        assert children_order in ["LR", "IO", "RL"], "unknown children order"

        self.children_order = children_order

    def guarantees_well_typedness(self) -> bool:
        return False

    def get_unconstrained_version(self) -> "GPUTransitionSystem":
        """
        Return an unconstrained version that does not do type checking.
        :return:
        """
        return self

    def _construct_seq(self, tree: Tree) -> List[Decision]:
        own_position = tree.node[0]
        push_actions = []
        recursive_actions = []

        if self.children_order == "LR":
            children = tree.children
        elif self.children_order == "RL":
            children = reversed(tree.children)
        elif self.children_order == "IO":
            left_part = []
            right_part = []
            for child in tree.children:
                if child.node[0] < own_position:
                    left_part.append(child)
                else:
                    right_part.append(child)
            children = list(reversed(left_part)) + right_part
        else:
            raise ValueError("Unknown children order: "+self.children_order)

        for child in children:
            if child.node[1].label == "IGNORE":
                continue

            push_actions.append(Decision(child.node[0], False, child.node[1].label, ("", ""), ""))
            recursive_actions.extend(self._construct_seq(child))

        if self.pop_with_0:
            relevant_position = 0
        else:
            relevant_position = own_position

        if self.reverse_push_actions:
            push_actions = list(reversed(push_actions))

        return push_actions + [Decision(relevant_position, True, "", (tree.node[1].fragment, tree.node[1].typ), tree.node[1].lexlabel)] + recursive_actions

    def get_order(self, sentence: AMSentence) -> Iterable[Decision]:
        t = Tree.from_am_sentence(sentence)
        r = [Decision(t.node[0], False, t.node[1].label, ("", ""), "")] + self._construct_seq(t)
        return r

    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        return all(x.head == y.head for x, y in zip(gold_sentence, predicted)) and \
               all(x.label == y.label for x, y in zip(gold_sentence, predicted)) and \
               all(x.fragment == y.fragment and x.typ == y.typ for x, y in zip(gold_sentence, predicted))


    def initial_state(self, sentence : AMSentence, decoder_state : Any) -> ParsingState:
        stack = [0]
        seen = set()
        substack = []
        heads = [0 for _ in range(len(sentence))]
        children = {i: [] for i in range(len(sentence) + 1)}
        labels = ["IGNORE" for _ in range(len(sentence))]
        lex_labels = ["_" for _ in range(len(sentence))]
        supertags = [("_","_") for _ in range(len(sentence))]

        return DFSChildrenFirstState(decoder_state, 0, 0.0, sentence, self.additional_lexicon, heads,
                                    children, labels, supertags, lex_labels, stack, seen, substack)

    def step(self, state : DFSChildrenFirstState, decision: Decision, in_place: bool = False) -> ParsingState:
        if in_place:
            copy = state
        else:
            copy = state.copy()
        copy.step += 1

        if state.stack:
            position = decision.position

            if (position in copy.seen and not self.pop_with_0) or \
                (position == 0 and self.pop_with_0) or copy.step == 2: #second step is always pop
                tos = copy.stack.pop()

                if not self.pop_with_0:
                    assert tos == position

                if copy.constants is not None and tos != 0:
                    copy.constants[tos - 1] = decision.supertag

                if copy.lex_labels is not None and tos != 0:
                    copy.lex_labels[tos - 1] = decision.lexlabel

                if self.reverse_push_actions:
                    copy.stack.extend(copy.substack)
                else:
                    copy.stack.extend(reversed(copy.substack))
                copy.substack = []
            else:
                copy.heads[position - 1] = copy.stack[-1]

                copy.children[copy.stack[-1]].append(position)  # 1-based

                copy.edge_labels[position - 1] = decision.label

                # push onto stack
                copy.substack.append(position)

            copy.seen.add(position)
            if copy.stack:
                copy.active_node = copy.stack[-1]
            else:
                copy.active_node = 0
        else:
            copy.active_node = 0
        copy.score = copy.score + decision.score
        return copy


