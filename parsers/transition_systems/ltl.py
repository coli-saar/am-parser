from copy import deepcopy
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch

from topdown_parser.am_algebra import AMType, new_amtypes
from topdown_parser.am_algebra.new_amtypes import ByApplySet, ModCache, ReadCache
from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.additional_lexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.transition_systems.ltf import typ2supertag, typ2i, collect_sources
from topdown_parser.transition_systems.parsing_state import ParsingState
from topdown_parser.transition_systems.transition_system import TransitionSystem
from .decision import Decision

from topdown_parser.transition_systems.utils import scores_to_selection, get_and_convert_to_numpy, get_best_constant, \
    single_score_to_selection, get_top_k_choices, copy_optional_set

import heapq

import numpy as np


class LTLState(ParsingState):

    def __init__(self, decoder_state: Any, active_node: int, score: float, sentence: AMSentence,
                 lexicon: AdditionalLexicon, heads: List[int], children: Dict[int, List[int]], edge_labels: List[str],
                 constants: List[Tuple[str, str]], lex_labels: List[str], stack: List[int], seen: Set[int], substack: List[int],
                 lexical_types: List[AMType], term_types : List[Set[AMType]], applysets_collected: List[Optional[Set[str]]], words_left : int, root_determined : bool,
                 sources_still_to_fill: List[int]):
        super().__init__(decoder_state, active_node, score, sentence, lexicon, heads, children, edge_labels, constants,
                         lex_labels, stack, seen)
        self.substack = substack
        self.lexical_types = lexical_types
        self.applysets_collected = applysets_collected
        self.words_left = words_left
        self.root_determined = root_determined
        self.term_types = term_types

        self.sources_still_to_fill = sources_still_to_fill
        self.step = 0

    def copy(self) -> "ParsingState":
        copy = LTLState(self.decoder_state, self.active_node, self.score, self.sentence,
                        self.lexicon, list(self.heads), deepcopy(self.children), list(self.edge_labels),
                        list(self.constants), list(self.lex_labels), list(self.stack), set(self.seen), list(self.substack),
                        list(self.lexical_types), copy_optional_set(self.term_types), copy_optional_set(self.applysets_collected), self.words_left, self.root_determined,
                        list(self.sources_still_to_fill))
        copy.step = self.step
        return copy


    def is_complete(self) -> bool:
        complete = self.stack == []

        if complete:
            assert sum(self.sources_still_to_fill) == 0
            assert self.substack == []

        return complete



class LTL(TransitionSystem):
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
        super().__init__(additional_lexicon)
        self.pop_with_0 = pop_with_0
        self.reverse_push_actions = reverse_push_actions
        assert children_order in ["LR", "IO", "RL"], "unknown children order"

        self.children_order = children_order

        self.typ2supertag : Dict[AMType, Set[int]] = typ2supertag(self.additional_lexicon)#which supertags have the given type?
        self.supertag2typ : Dict[int, AMType] = dict()

        for typ, constants in self.typ2supertag.items():
            for constant in constants:
                self.supertag2typ[constant] = typ

        self.typ2i :  Dict[AMType, int] = typ2i(self.additional_lexicon) # which type has what id?

        self.candidate_lex_types = new_amtypes.CandidateLexType({typ for typ in self.typ2i.keys()})

        self.sources: Set[str] = collect_sources(self.additional_lexicon)

        modify_sources = {source for source in self.sources if self.additional_lexicon.contains("edge_labels", "MOD_"+source) }
        self.modify_ids = {self.additional_lexicon.get_id("edge_labels", "MOD_"+source) for source in modify_sources} #ids of modify edges
        self.mod_cache = ModCache(self.typ2i.keys())

        self.apply_cache = ByApplySet(self.typ2i.keys())

        self.read_cache = ReadCache()

    def predict_supertag_from_tos(self) -> bool:
        return True

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
        constants = [("_","_") for _ in range(len(sentence))]
        lexical_types = [AMType.parse_str("_") for _ in range(len(sentence))]
        term_types = [None for _ in range(len(sentence))]
        applysets_collected = [None for _ in range(len(sentence))]

        return LTLState(decoder_state, 0, 0.0, sentence,
                 self.additional_lexicon, heads, children, labels,
                 constants, lex_labels, stack, seen, substack,
                 lexical_types, term_types, applysets_collected, len(sentence), False, [0 for _ in sentence])

    def step(self, state: LTLState, decision: Decision, in_place: bool = False) -> ParsingState:
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

                if tos != 0:
                    copy.constants[tos - 1] = decision.supertag
                    lexical_type_tos = self.read_cache.parse_str(decision.supertag[1])
                    copy.lexical_types[tos - 1] = lexical_type_tos
                    copy.lex_labels[tos - 1] = decision.lexlabel

                    #now determine term types of children
                    for child_id in state.children[tos]: # 1-based children
                        child_id -= 1 # 0-based children
                        label = copy.edge_labels[child_id]

                        copy.applysets_collected[child_id] = set()

                        if label.startswith("APP_"):
                            # get request at source
                            source = label.split("_")[1]
                            req = lexical_type_tos.get_request(source)
                            copy.term_types[child_id] = {req}

                        elif label.startswith("MOD_"):
                            source = label.split("_")[1]
                            copy.term_types[child_id] = set(self.mod_cache.get_modifiers_with_source(lexical_type_tos, source))
                        else:
                            raise ValueError("Somehow the invalid edge label "+label+" was produced")

                if self.reverse_push_actions:
                    copy.stack.extend(copy.substack)
                else:
                    copy.stack.extend(reversed(copy.substack))
                copy.substack = []
            else:
                tos = copy.stack[-1]
                copy.heads[position - 1] = tos

                assert position <= len(copy.sentence)
                copy.children[tos].append(position)  # 1-based

                copy.edge_labels[position - 1] = decision.label

                if decision.label.startswith("APP_"):
                    source = decision.label.split("_")[1]
                    copy.applysets_collected[copy.active_node-1].add(source)
                    smallest_apply_set = np.inf
                    for term_typ in copy.term_types[tos-1]:
                        for lexical_type, apply_set in self.candidate_lex_types.get_candidates_with_apply_set(term_typ,
                                copy.applysets_collected[tos-1], copy.words_left + len(state.applysets_collected[tos-1])):

                            rest_of_apply_set = apply_set - copy.applysets_collected[tos-1]
                            smallest_apply_set = min(smallest_apply_set, len(rest_of_apply_set))

                    assert smallest_apply_set < np.inf
                    copy.sources_still_to_fill[tos-1] = smallest_apply_set

                elif decision.label == "ROOT" and not copy.root_determined:
                    copy.term_types[position-1] = {AMType.parse_str("()")}
                    copy.applysets_collected[position-1] = set()
                    copy.root_determined = True

                copy.words_left -= 1

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

    def make_decision(self, scores: Dict[str, torch.Tensor], state : LTLState) -> Decision:
        # Select node:
        child_scores = scores["children_scores"].detach().cpu() # shape (input_seq_len)
        INF = 10e10

        if not state.root_determined: # First decision must choose root.
            child_scores[0] = -INF
            s, selected_node = torch.max(child_scores, dim=0)
            return Decision(int(selected_node), False, "ROOT", ("",""), "",termtyp=None, score=float(s))

        #Cannot select nodes that we have visited already.
        for seen in state.seen:
            if self.pop_with_0 and seen == 0:
                pass
            elif not self.pop_with_0 and seen == state.active_node:
                pass
            else:
                child_scores[seen] = -INF

        if state.active_node != 0 and state.sources_still_to_fill[state.active_node-1] > 0:
            # Cannot close the current node if the smallest apply set reachable from the active node requires is still do add more APP edges.
            if self.pop_with_0:
                child_scores[0] = -INF
            else:
                child_scores[state.active_node] = -INF

        score = 0.0
        s, selected_node = torch.max(child_scores, dim=0)
        score += s

        if state.step == 1 or state.active_node == 0:
            #we are done (or after first step), do nothing.
            return Decision(0, False, "", ("",""), "", score=0.0)

        if (selected_node in state.seen and not self.pop_with_0) or (selected_node == 0 and self.pop_with_0):
            # pop node, select constant and lexical label.
            constant_scores = scores["constants_scores"].cpu().numpy()
            #max_score = -np.inf
            #best_constant = None
            possible_constants = set()
            for term_typ in state.term_types[state.active_node-1]:
                possible_lex_types = self.apply_cache.by_apply_set(term_typ, frozenset(state.applysets_collected[state.active_node-1]))
                for lex_type in possible_lex_types:
                    possible_constants.update(self.typ2supertag[lex_type])

            assert len(possible_constants) > 0
            best_constant, max_score = get_best_constant(possible_constants, constant_scores)
            pop_node = 0 if self.pop_with_0 else state.active_node
            selected_lex_label = self.additional_lexicon.get_str_repr("lex_labels", int(scores["lex_labels"].cpu().numpy()))
            score += s
            return Decision(pop_node, True, "", AMSentence.split_supertag(self.additional_lexicon.get_str_repr("constants", best_constant)), selected_lex_label, score=score)

        # APP or MOD?
        label_scores = scores["all_labels_scores"][selected_node].cpu().numpy() #shape (edge vocab size)

        max_apply_score = -np.inf
        #best_apply_source = None
        #best_lex_type = None # for debugging purposes
        smallest_apply_set = state.sources_still_to_fill[state.active_node - 1]

        apply_of_tos = state.applysets_collected[state.active_node-1]
        possible_sources = set()
        for term_typ in state.term_types[state.active_node-1]:
            for lexical_type, apply_set in self.candidate_lex_types.get_candidates_with_apply_set(term_typ, apply_of_tos, state.words_left + len(apply_of_tos)):
                rest_of_apply_set = apply_set - apply_of_tos

                if len(rest_of_apply_set) <= state.words_left:
                    possible_sources.update(rest_of_apply_set)

        best_apply_edge_id = None
        if len(possible_sources) > 0:
            edge_ids = {self.additional_lexicon.get_id("edge_labels", "APP_"+source) for source in possible_sources}
            best_apply_edge_id, max_apply_score = get_best_constant(edge_ids, label_scores)

        # Check MODIFY
        max_modify_score = -np.inf
        best_modify_edge_id = None
        if state.words_left - smallest_apply_set > 0:
            best_modify_edge_id, max_modify_score = get_best_constant(self.modify_ids, label_scores)

        # Apply our choice
        if max_modify_score > max_apply_score:
            # MOD
            return Decision(int(selected_node), False, self.additional_lexicon.get_str_repr("edge_labels",  best_modify_edge_id), ("",""),"", score=score+max_modify_score)
        elif max_apply_score > -np.inf:
            # APP
            return Decision(int(selected_node), False, self.additional_lexicon.get_str_repr("edge_labels",  best_apply_edge_id), ("",""),"", score=score+max_apply_score)
        else:
            raise ValueError("Could not select action. Bug.")


    def top_k_decision(self, scores: Dict[str, torch.Tensor], state : LTLState, k : int) -> List[Decision]:
        # Select node:
        child_scores = scores["children_scores"].cpu() # shape (input_seq_len)
        #Cannot select nodes that we have visited already (except if not pop with 0 and currently active, then we can close).
        INF = 10e10
        forbidden = 0
        for seen in state.seen:
            if self.pop_with_0 and seen == 0:
                pass
            elif not self.pop_with_0 and seen == state.active_node:
                pass
            else:
                child_scores[seen] = -INF
                forbidden += 1

        if state.active_node != 0 and state.sources_still_to_fill[state.active_node-1] > 0:
            # Cannot close the current node if the smallest apply set reachable from the active node requires is still do add more APP edges.
            if self.pop_with_0:
                child_scores[0] = -INF
            else:
                child_scores[state.active_node] = -INF
            forbidden += 1

        if not state.root_determined: # First decision must choose root.
            child_scores[0] = -INF
            forbidden += 1

        at_most_k = min(k, len(state.sentence)+1-forbidden) #don't let beam search explore things that are not well-formed.
        children_scores, children = torch.sort(child_scores, descending=True)
        children_scores = children_scores[:at_most_k] #shape (at_most_k)
        children = children[:at_most_k] #shape (at_most_k)
        # Now have k best children

        label_scores = scores["all_labels_scores"][children] # (at_most_k, label vocab size)

        children = children.cpu().numpy()
        children_scores = children_scores.cpu().numpy()
        label_scores = label_scores.cpu().numpy()
        constant_scores = scores["constants_scores"].cpu().numpy()
        #lex_label_score, selected_lex_label = single_score_to_selection(scores, self.additional_lexicon, "lex_labels")
        selected_lex_label = self.additional_lexicon.get_str_repr("lex_labels", int(scores["lex_labels"].cpu().numpy()))

        decisions = []
        for selected_node, node_score, label_scores in zip(children, children_scores, label_scores):
            assert selected_node.shape == ()
            assert label_scores.shape == (self.additional_lexicon.vocab_size("edge_labels"),)

            if not state.root_determined:
                decisions.append(Decision(int(selected_node), False, "ROOT", ("",""), "",termtyp=None, score=node_score))
                continue

            if state.step == 1 or state.active_node == 0:
                #we are done (or after first step), do nothing.
                decisions.append(Decision(0, False, "", ("",""), "", score=0.0))
                break

            if (selected_node in state.seen and not self.pop_with_0) or (selected_node == 0 and self.pop_with_0):
                # pop node, select constant and lexical label.
                pop_node = 0 if self.pop_with_0 else state.active_node
                possible_lex_types = set()
                for term_typ in state.term_types[state.active_node-1]:
                    possible_lex_types.update(self.apply_cache.by_apply_set(term_typ, frozenset(state.applysets_collected[state.active_node-1])))

                assert len(possible_lex_types) > 0
                for lex_type in possible_lex_types:
                    constant, constant_score = get_best_constant(self.typ2supertag[lex_type], constant_scores)
                    decisions.append(Decision(pop_node, True, "", AMSentence.split_supertag(self.additional_lexicon.get_str_repr("constants", constant)),
                                                   selected_lex_label, score=constant_score + node_score))

                # for term_typ in state.term_types[state.active_node-1]:
                #     possible_lex_types = self.apply_cache.by_apply_set(term_typ, frozenset(state.applysets_collected[state.active_node-1]))
                #     if possible_lex_types:
                #         possible_constants = {constant for lex_type in possible_lex_types for constant in self.typ2supertag[lex_type]}
                #         constant, constant_score = get_best_constant(possible_constants, constant_scores)
                #         decisions.append(Decision(pop_node, "", AMSentence.split_supertag(self.additional_lexicon.get_str_repr("constants", constant)),
                #                                   selected_lex_label, score=constant_score + node_score))
                continue

            smallest_apply_set = state.sources_still_to_fill[state.active_node - 1]

            apply_of_tos = state.applysets_collected[state.active_node-1]

            # APP
            possible_sources = set()
            for term_typ in state.term_types[state.active_node-1]:
                for lexical_type, apply_set in self.candidate_lex_types.get_candidates_with_apply_set(term_typ, apply_of_tos, state.words_left + len(apply_of_tos)):
                    rest_of_apply_set = apply_set - apply_of_tos

                    if len(rest_of_apply_set) <= state.words_left:
                        possible_sources.update(rest_of_apply_set)

            apply_ids = {self.additional_lexicon.get_id("edge_labels", "APP_"+source) for source in possible_sources}
            for edge_id, apply_score in get_top_k_choices(apply_ids, label_scores, k):
                decisions.append(Decision(int(selected_node), False, self.additional_lexicon.get_str_repr("edge_labels", edge_id), ("",""), "", score = node_score+apply_score))

            # MOD
            if state.words_left - smallest_apply_set > 0:
                for edge_id, modify_score in get_top_k_choices(self.modify_ids, label_scores, k):
                    decisions.append(Decision(int(selected_node), False, self.additional_lexicon.get_str_repr("edge_labels", edge_id), ("",""), "", score = node_score+modify_score))

        return decisions


    def assumes_greedy_ok(self) -> Set[str]:
        """
        The dictionary keys of the context provider which we make greedy decisions on in top_k_decisions
        because we assume these choices won't impact future scores.
        :return:
        """
        return set()