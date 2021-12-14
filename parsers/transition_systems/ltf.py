# cython: language_level=3
import heapq
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch
from allennlp.common.checks import ConfigurationError

from topdown_parser.am_algebra import AMType, NonAMTypeException, new_amtypes
from topdown_parser.am_algebra.new_amtypes import CandidateLexType, ModCache, ReadCache
from topdown_parser.am_algebra.tools import get_term_types
from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.additional_lexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.edge_label_model import EdgeLabelModel
from topdown_parser.nn.utils import get_device_id
from topdown_parser.transition_systems import utils
from topdown_parser.transition_systems.parsing_state import ParsingState
from topdown_parser.transition_systems.transition_system import TransitionSystem
from .decision import Decision
from topdown_parser.transition_systems.utils import scores_to_selection, get_best_constant, single_score_to_selection, \
    is_empty, get_top_k_choices, copy_optional_set

import numpy as np

from .dfs import DFS


def typ2supertag(lexicon : AdditionalLexicon) -> Dict[AMType, Set[int]]:
    _typ2supertag : Dict[AMType, Set[int]] = dict() #which supertags have the given type?

    for supertag, i in lexicon.sublexica["constants"]:
        _, typ = AMSentence.split_supertag(supertag)
        try:
            typ = AMType.parse_str(typ)
            if typ not in _typ2supertag:
                _typ2supertag[typ] = set()

            _typ2supertag[typ].add(i)
        except NonAMTypeException:
            print("Skipping type", typ)
    return _typ2supertag

def typ2i(additional_lexicon : AdditionalLexicon) -> Dict[AMType, int]:
    _typ2i :  Dict[AMType, int] = dict()
    for typ, i in additional_lexicon.sublexica["term_types"]:
        try:
            _typ2i[AMType.parse_str(typ)] = i
        except NonAMTypeException:
            pass
    return _typ2i

def collect_sources(additional_lexicon : AdditionalLexicon) -> Set[str]:
    sources: Set[str] = set()
    for label, _ in additional_lexicon.sublexica["edge_labels"]:
        if "_" in label:
            sources.add(label.split("_")[1])
    return sources

class LTFState(ParsingState):

    def __init__(self, decoder_state: Any, active_node: int, score: float, sentence: AMSentence,
                 lexicon: AdditionalLexicon, heads: List[int], children: Dict[int, List[int]], edge_labels: List[str],
                 constants: List[Tuple[str, str]], lex_labels: List[str], stack: List[int], seen: Set[int],
                 lexical_types: List[AMType], term_types : List[Set[AMType]], applysets_todo: List[Optional[Set[str]]], words_left : int, root_determined : bool):
        super().__init__(decoder_state, active_node, score, sentence, lexicon, heads, children, edge_labels, constants,
                         lex_labels, stack, seen)
        self.lexical_types = lexical_types
        self.applysets_todo = applysets_todo
        self.words_left = words_left
        self.root_determined = root_determined
        self.term_types = term_types

    def copy(self) -> "ParsingState":
        return LTFState(self.decoder_state, self.active_node, self.score, self.sentence,
                        self.lexicon, list(self.heads), {node: list(children) for node, children in self.children.items()}, list(self.edge_labels),
                        list(self.constants), list(self.lex_labels), list(self.stack), set(self.seen),
                        list(self.lexical_types), [set(s) for s in self.term_types], copy_optional_set(self.applysets_todo), self.words_left, self.root_determined)

    def sources_to_be_filled(self) -> int:
        return sum(len(a) if a is not None else 0 for a in self.applysets_todo)

    def is_complete(self) -> bool:
        complete = self.stack == []

        if complete:
            assert self.sources_to_be_filled() == 0

        return complete

@TransitionSystem.register("ltf")
class LTF(TransitionSystem):
    """
    Lexical type first strategy.
    """

    def __init__(self, children_order: str, pop_with_0: bool, additional_lexicon : AdditionalLexicon):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        """
        super().__init__(additional_lexicon)
        self.pop_with_0 = pop_with_0
        assert children_order in ["LR", "IO"], "unknown children order"

        self.children_order = children_order

        self.typ2supertag : Dict[AMType, Set[int]] = typ2supertag(self.additional_lexicon)#which supertags have the given type?

        self.typ2i :  Dict[AMType, int] = typ2i(self.additional_lexicon) # which type has what id?

        self.candidate_lex_types = new_amtypes.CandidateLexType({typ for typ in self.typ2i.keys()})

        #self.subtype_cache = SubtypeCache(self.typ2i.keys())
        self.mod_cache = ModCache(self.typ2i.keys())

        self.sources: Set[str] = collect_sources(self.additional_lexicon)
        modify_sources = {source for source in self.sources if self.additional_lexicon.contains("edge_labels", "MOD_"+source) }
        self.modify_ids = {self.additional_lexicon.get_id("edge_labels", "MOD_"+source) for source in modify_sources} #ids of modify edges

        self.read_cache = ReadCache()

    def predict_supertag_from_tos(self) -> bool:
        return True

    def guarantees_well_typedness(self) -> bool:
        return True

    def get_unconstrained_version(self) -> "TransitionSystem":
        return DFS(self.children_order, self.pop_with_0, self.additional_lexicon)

    def validate_model(self, parser : "TopDownDependencyParser") -> None:
        """
        Check if the parsing model produces all the scores that we need.
        :param parser:
        :return:
        """
        if parser.term_type_tagger is None:
            raise ConfigurationError("ltf transition system requires term tagger")

        if parser.lex_label_tagger is None:
            raise ConfigurationError("ltf transition system requires lex label tagger")

        if parser.supertagger is None:
            raise ConfigurationError("ltf transition system requires tagger for constants")

    def get_order(self, sentence: AMSentence) -> Iterable[Decision]:
        t = Tree.from_am_sentence(sentence)
        term_types = get_term_types(sentence)

        def _construct_seq(tree: Tree, is_first_child : bool, parent_type : Tuple[str, str], parent_lex_label : str, parent_term_type : AMType) -> List[Decision]:
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
                beginning = [Decision(own_position, False, tree.node[1].label, parent_type, parent_lex_label, parent_term_type)]
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
                ret.extend(_construct_seq(child, i == 0, (tree.node[1].fragment, tree.node[1].typ), tree.node[1].lexlabel, term_types[own_position-1]))

            last_position = 0 if self.pop_with_0 else own_position
            if len(tree.children) == 0:
                #This subtree has no children, thus also no first child at which we would determine the type of the parent
                #Let's determine the type now.
                last_decision = Decision(last_position, True, "", (tree.node[1].fragment, tree.node[1].typ),
                                         tree.node[1].lexlabel, term_types[own_position-1])
            else:
                last_decision = Decision(last_position, True, "", ("",""), "")
            ret.append(last_decision)
            return ret

        return _construct_seq(t, False, ("",""),"", AMType.parse_str("_"))

    def initial_state(self, sentence : AMSentence, decoder_state : Any) -> ParsingState:
        stack = [0]
        seen = set()
        heads = [0 for _ in range(len(sentence))]
        children = {i: [] for i in range(len(sentence) + 1)}
        labels = ["IGNORE" for _ in range(len(sentence))]
        lex_labels = ["_" for _ in range(len(sentence))]
        supertags = [("_","_") for _ in range(len(sentence))]
        lexical_types = [AMType.parse_str("_") for _ in range(len(sentence))]
        term_types = [set() for _ in range(len(sentence))]
        applysets_todo = [None for _ in range(len(sentence))]

        return LTFState(decoder_state, 0, 0.0, sentence,
                        self.additional_lexicon, heads, children, labels,
                        supertags, lex_labels, stack, seen,
                        lexical_types, term_types, applysets_todo, len(sentence), False)

    def step(self, state: LTFState, decision: Decision, in_place: bool = False) -> ParsingState:
        if in_place:
            copy = state
        else:
            copy = state.copy()

        if state.stack:
            if copy.constants[state.active_node-1] == ("_","_") and state.active_node != 0:
                # first time that state.active_node has become active.
                copy.constants[state.active_node-1] = decision.supertag
                copy.lex_labels[state.active_node-1] = decision.lexlabel

                # Determine apply set which we have to fulfill.
                lex_type = self.read_cache.parse_str(decision.supertag[1])
                copy.lexical_types[state.active_node-1] = lex_type
                term_type = decision.termtyp
                applyset = lex_type.get_apply_set(term_type)
                assert term_type in copy.term_types[state.active_node-1]

                copy.term_types[state.active_node-1] = {term_type}

                copy.applysets_todo[state.active_node-1] = applyset

            if decision.position == 0 and self.pop_with_0:
                copy.stack.pop()
            elif not self.pop_with_0 and decision.position in copy.seen:
                popped = copy.stack.pop()
                assert popped == decision.position
            else:
                copy.heads[decision.position-1] = copy.stack[-1]
                copy.words_left -= 1 # one word gets attached.

                copy.children[copy.stack[-1]].append(decision.position)  # 1-based

                copy.edge_labels[decision.position - 1] = decision.label

                tos_lexical_type = copy.lexical_types[copy.stack[-1]-1]
                if decision.label.startswith("APP_"):
                    source = decision.label.split("_")[1]
                    copy.applysets_todo[copy.stack[-1]-1].remove(source) #remove obligation to fill source.

                    copy.term_types[decision.position-1] = {tos_lexical_type.get_request(source)}

                elif decision.label.startswith("MOD_"):
                    source = decision.label.split("_")[1]
                    copy.term_types[decision.position-1] = set(self.mod_cache.get_modifiers_with_source(tos_lexical_type, source)) #TODO speed improvement?

                elif decision.label == "ROOT" and not copy.root_determined:
                    copy.term_types[decision.position-1] = {AMType.parse_str("()")}
                else:
                    raise ValueError("Edge label "+decision.label+" not allowed here.")

                # push onto stack
                copy.stack.append(decision.position)

            copy.seen.add(decision.position)

            if not copy.stack:
                copy.active_node = 0
            else:
                copy.active_node = copy.stack[-1]
        else:
            copy.active_node = 0

        copy.root_determined = True
        copy.score = copy.score + decision.score
        return copy

    def make_decision(self, scores: Dict[str, torch.Tensor], state : LTFState) -> Decision:
        # Select node:
        child_scores = scores["children_scores"].detach().cpu() # shape (input_seq_len)
        #Cannot select nodes that we have visited already.
        nINF = -10e10

        for seen in state.seen:
            if self.pop_with_0 and seen == 0:
                pass
            elif not self.pop_with_0 and seen == state.active_node:
                pass
            else:
                child_scores[seen] = nINF

        if not state.root_determined: # First decision must choose root.
            child_scores[0] = nINF
            s, selected_node = torch.max(child_scores, dim=0)
            return Decision(int(selected_node), False, "ROOT", ("",""), "",termtyp=None, score=float(s))

        if state.active_node == 0:
            return Decision(0, False, "", ("",""), "", termtyp=None, score=0.0)

        score = 0.0
        constant_scores = scores["constants_scores"].cpu().numpy()
        term_type_scores = scores["term_types_scores"].cpu().numpy()

        selected_constant = ("","")
        selected_term_type = None
        selected_lex_label = "_"

        lexical_type_of_tos = state.lexical_types[state.active_node-1]
        applyset_todo_tos = state.applysets_todo[state.active_node-1]

        sources_to_be_filled = state.sources_to_be_filled()

        # Greedily choose best constant, if needed at this point.
        if state.constants[state.active_node-1] == ("_","_"):
            # active node needs to get a lexical type, choose one.
            selected_lex_label = self.additional_lexicon.get_str_repr("lex_labels", int(scores["lex_labels"].cpu().numpy()))

            possible_term_types = state.term_types[state.active_node-1]
            max_constant_score = -np.inf
            best_constant = None
            best_term_type = None

            for term_type in possible_term_types:
                local_term_type_score = term_type_scores[self.typ2i[term_type]]
                for lex_type in self.candidate_lex_types.get_candidates(term_type, state.words_left - sources_to_be_filled):
                    best_local_constant, best_local_constant_score = get_best_constant(self.typ2supertag[lex_type], constant_scores)
                    local_decision_score = best_local_constant_score + local_term_type_score

                    if local_decision_score >= max_constant_score:
                        max_constant_score = local_decision_score
                        best_constant = best_local_constant
                        best_term_type = term_type

            assert best_constant is not None #we have to be able to find something here!
            selected_constant = AMSentence.split_supertag(self.additional_lexicon.get_str_repr("constants", best_constant))
            lexical_type_of_tos = self.read_cache.parse_str(selected_constant[1])
            selected_term_type = best_term_type
            applyset_todo_tos = lexical_type_of_tos.get_apply_set(selected_term_type)
            score += max_constant_score
            sources_to_be_filled += len(applyset_todo_tos)

        assert applyset_todo_tos is not None

        #Check if we must not close the current node
        if applyset_todo_tos is not None and len(applyset_todo_tos) > 0:
            if self.pop_with_0:
                child_scores[0] = nINF
            else:
                child_scores[state.active_node] = nINF
        elif applyset_todo_tos is not None and len(applyset_todo_tos) == 0 and state.words_left - sources_to_be_filled == 0:
            # somewhere in the tree (but not here!) there are sources to fill
            # the number of words left exactly matches that. Since we don't have to fill a source here, we must pop!
            if self.pop_with_0:
                child_scores[0] = 10e10
            else:
                child_scores[state.active_node] = 10e10

        s, selected_node = torch.max(child_scores, dim=0)
        score += s

        if (selected_node == 0 and self.pop_with_0) or (selected_node == state.active_node and not self.pop_with_0):
            return Decision(int(selected_node), True, "", selected_constant, selected_lex_label, selected_term_type, score=score)

        words_left_after_this = state.words_left - 1

        label_scores = scores["all_labels_scores"][selected_node].cpu().numpy() #shape (edge vocab size)

        #Check if we want to do APP or MOD
        max_apply_score = -np.inf
        best_apply_source = None

        for todo_source in applyset_todo_tos:
            source_score = label_scores[self.additional_lexicon.get_id("edge_labels", "APP_"+todo_source)]
            # TODO what if APP_todo_source is not a valid edge label? right now we get the UNK score.
            if source_score >= max_apply_score:
                best_apply_source = todo_source
                max_apply_score = source_score

        # What if we want to modify? Find best mod operation.
        max_mod_score = -np.inf
        best_modify_edge = None

        if words_left_after_this - sources_to_be_filled >= 0:
            #MOD is allowed, we allow any MOD_m modification because we can always choose a term type that fits in the next step.
            best_modify_edge, max_mod_score = get_best_constant(self.modify_ids, label_scores)

        if max_apply_score > max_mod_score:
            return Decision(int(selected_node), False, "APP_"+best_apply_source, selected_constant, selected_lex_label, selected_term_type, score=score+max_apply_score)
        elif max_mod_score > -np.inf:
            return Decision(int(selected_node), False, self.additional_lexicon.get_str_repr("edge_labels", best_modify_edge), selected_constant, selected_lex_label, selected_term_type, score=score+max_mod_score)
        else:
            raise ValueError("Could not perform any action. Bug.")

    def top_k_lexical_types(self, constant_scores : np.array, term_type_scores : np.array, state : LTFState, k : int) -> List[Tuple[AMType, AMType, int, Set[str], float]]:
        typing_info = []
        if state.constants[state.active_node-1] == ("_","_") and state.active_node != 0:
            # active node needs to get a lexical type, choose one.

            possible_term_types = state.term_types[state.active_node-1]
            assert len(possible_term_types) > 0

            sources_to_be_filled = state.sources_to_be_filled()

            for term_type in possible_term_types:
                local_term_type_score = term_type_scores[self.typ2i[term_type]]
                for lex_type in self.candidate_lex_types.get_candidates(term_type, state.words_left - sources_to_be_filled):
                    best_local_constant, best_local_constant_score = get_best_constant(self.typ2supertag[lex_type], constant_scores)
                    local_decision_score = best_local_constant_score + local_term_type_score
                    applyset_todo_tos = lex_type.get_apply_set(term_type)

                    typing_info.append((lex_type, term_type, best_local_constant, applyset_todo_tos, local_decision_score))
            assert len(typing_info) > 0

        return heapq.nlargest(k, typing_info, key=lambda tupl: tupl[-1])

    def top_k_decision(self, scores: Dict[str, torch.Tensor], state: LTFState, k : int) -> List[Decision]:

        if k == 1:
            raise ValueError("This currently doesn't work for k=1 because the locally best lexical type doesn't "
                             "necessarily fit to the locally best child")

        if state.root_determined and state.active_node == 0:
            return [Decision(0, False, "", ("",""), "", termtyp=None, score=0.0)]


        # Find best constants, if we have to choose them:
        head_types = self.top_k_lexical_types(scores["constants_scores"].cpu().numpy(), scores["term_types_scores"].cpu().numpy(), state, k)
        determine_head_type = True
        if not head_types:
            determine_head_type = False
            if not state.root_determined: # fill in dummy values
                head_types = [(None, None, 0, None, 0.0)]
            else:
                head_types = [(state.lexical_types[state.active_node-1], None, 0, state.applysets_todo[state.active_node-1], 0.0)]

        # Select node:
        child_scores = scores["children_scores"] # shape (input_seq_len)
        pop_node = 0 if self.pop_with_0 else state.active_node
        #Cannot select nodes that we have visited already
        forbidden = 0
        INF = 10e10
        for seen in state.seen:
            if seen != pop_node:
                child_scores[seen] = -INF
                forbidden += 1

        at_most_k = min(k, len(state.sentence)+1-forbidden) #don't let beam search explore things that are not well-formed.
        children_scores, children = torch.sort(child_scores, descending=True)
        children_scores = children_scores[:at_most_k] #shape (at_most_k)
        children = children[:at_most_k] #shape (at_most_k)

        device = get_device_id(children)

        #Add pop node, we might have to do this, depending on the lexical type
        if pop_node not in children:
            children = torch.cat((children, torch.from_numpy(np.array([pop_node])).to(device)))
            children_scores = torch.cat((children_scores, child_scores[pop_node].unsqueeze(0)))
            at_most_k += 1

        # Now we have k best children

        label_scores = scores["all_labels_scores"][children] #(at_most_k, label vocab_size)

        children = children.cpu().numpy()
        children_scores = children_scores.cpu().numpy()
        label_scores = label_scores.cpu().numpy()

        selected_lex_label = self.additional_lexicon.get_str_repr("lex_labels", int(scores["lex_labels"].cpu().numpy()))

        decisions = []

        for lexical_type_of_tos, term_type_of_tos, best_local_constant, applyset_todo_tos, local_decision_score in head_types:

            sources_to_be_filled = state.sources_to_be_filled()

            if determine_head_type:
                # if we have to decide for a lexical type of our head as well, these go to our todolist as well.
                sources_to_be_filled += len(applyset_todo_tos)

            for selected_node, node_score, label_scores_this_node in zip(children, children_scores, label_scores):

                if not state.root_determined:
                    if selected_node != 0: # First decision must choose root.
                        decisions.append(Decision(int(selected_node),False, "ROOT", ("",""), "",termtyp=None, score=float(node_score)))
                    continue

                #Check if we must not close the current node
                if applyset_todo_tos is not None and len(applyset_todo_tos) > 0:
                    if selected_node == pop_node: # This won't work, skip
                        continue
                elif applyset_todo_tos is not None and len(applyset_todo_tos) == 0 and state.words_left - sources_to_be_filled == 0:
                    # somewhere in the tree (but not here!) there are sources to fill
                    # the number of words left exactly matches that. Since we don't have to fill a source here, we must pop!
                    if selected_node != pop_node: # This won't work, skip
                        continue

                if selected_node == pop_node:
                    if determine_head_type:
                        decisions.append(Decision(pop_node, True, "",
                                                  AMSentence.split_supertag(self.additional_lexicon.get_str_repr("constants", best_local_constant)),
                                                  selected_lex_label, term_type_of_tos, score=node_score + local_decision_score))
                    else:
                        decisions.append(Decision(pop_node, True, "", ("",""), "_", None, score=node_score))
                    continue

                words_left_after_this = state.words_left - 1

                # possible_sources = set()
                source_to_score = dict()

                for todo_source in applyset_todo_tos:
                    source_score = label_scores_this_node[self.additional_lexicon.get_id("edge_labels", "APP_"+todo_source)]
                    source_to_score[todo_source] = source_score

                top_k_mod_choices = get_top_k_choices(self.modify_ids, label_scores_this_node, k)

                if determine_head_type:
                    head_constant = AMSentence.split_supertag(self.additional_lexicon.get_str_repr("constants", best_local_constant))
                    for source in sorted(applyset_todo_tos, key=lambda source: source_to_score[source], reverse=True)[:k]:
                        decisions.append(Decision(int(selected_node), False, "APP_"+source, head_constant, selected_lex_label, term_type_of_tos,
                                                  score=node_score + local_decision_score + source_to_score[source]))

                    if words_left_after_this - sources_to_be_filled >= 0:
                        for edge_id, modify_score in top_k_mod_choices:
                            decisions.append(Decision(int(selected_node), False, self.additional_lexicon.get_str_repr("edge_labels", edge_id), head_constant, selected_lex_label, term_type_of_tos,
                                                      score=node_score + local_decision_score + modify_score))

                else:
                    for source in sorted(applyset_todo_tos, key=lambda source: source_to_score[source], reverse=True)[:k]:
                        decisions.append(Decision(int(selected_node), False, "APP_"+source, ("",""), "_", None,
                                                  score=node_score + source_to_score[source]))

                    if words_left_after_this - sources_to_be_filled >= 0:
                        for edge_id, modify_score in top_k_mod_choices:
                            decisions.append(Decision(int(selected_node), False, self.additional_lexicon.get_str_repr("edge_labels", edge_id), ("",""), "_", None,
                                                      score=node_score + modify_score))

        assert len(decisions) > 0
        return decisions

    def assumes_greedy_ok(self) -> Set[str]:
        return set()


    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        return all(x.head == y.head for x, y in zip(gold_sentence, predicted)) and \
               all(x.label == y.label for x, y in zip(gold_sentence, predicted)) and \
                all(x.fragment == y.fragment and x.typ == y.typ for x, y in zip(gold_sentence, predicted))
