from typing import List, Iterable, Optional, Any, Dict, Set, Tuple

import numpy as np
import torch

from parsers.am_algebra import AMType
from parsers.am_algebra.new_amtypes import ModCache
from parsers.am_algebra.tree import Tree
from parsers.dataset_readers.additional_lexicon import AdditionalLexicon
from parsers.dataset_readers.amconll_tools import AMSentence
from parsers.transition_systems.decision import DecisionBatch
from parsers.transition_systems.gpu_parsing.datastructures.list_of_list import BatchedListofList
from parsers.transition_systems.gpu_parsing.datastructures.stack import BatchedStack
from parsers.nn.utils import get_device_id
from parsers.transition_systems.gpu_parsing.dfs_children_first import GPUDFSChildrenFirst
from parsers.transition_systems.gpu_parsing.logic_torch import index_or, \
    make_bool_multipliable, are_eq
from parsers.transition_systems.batched_parsing_state import BatchedParsingState
from parsers.transition_systems.ltl import LTL
from parsers.transition_systems.transition_system import TransitionSystem


class GPULTLState(BatchedParsingState):

    def __init__(self,  decoder_state: Any,
                sentences: List[AMSentence],
                stack: BatchedStack,
                children: BatchedListofList,
                heads: torch.Tensor,
                edge_labels: torch.Tensor,
                constants: torch.Tensor,
                lex_labels: torch.Tensor,
                lexicon: AdditionalLexicon,
                lex_types : torch.Tensor,
                applyset : torch.Tensor
                 ):
        """

        :param decoder_state:
        :param sentences:
        :param stack:
        :param children:
        :param heads:
        :param edge_labels:
        :param constants:
        :param lex_labels:
        :param lexicon:
        :param lex_types: shape (batch_size, sent length)
        :param applyset: shape (batch_size, sent length, number of sources), belongs to currently active nodes
        """
        super(GPULTLState, self).__init__(decoder_state, sentences, stack, children, heads, edge_labels, constants, None, lex_labels, lexicon)
        self.lex_types = lex_types
        self.applyset = applyset
        self.step = 0
        self.w_c = self.get_lengths().clone()


#@GPUTransitionSystem.register("ltl")
@TransitionSystem.register("ltl")
class GPULTL(LTL):
    """
    DFS where when a node is visited for the second time, all its children are visited once.
    Afterwards, the first child is visited for the second time. Then the second child etc.
    """

    def __init__(self, children_order: str, pop_with_0: bool,
                 additional_lexicon: AdditionalLexicon,
                 reverse_push_actions: bool = False,
                 enable_assert : bool = False):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        reverse_push_actions means that the order of push actions is the opposite order in which the children of
        the node are recursively visited.
        """
        super().__init__(children_order, pop_with_0, additional_lexicon, reverse_push_actions)
        self.enable_assert = enable_assert

        self.i2source = sorted({label.split("_")[1] for label, _ in self.additional_lexicon.sublexica["edge_labels"] if "_" in label})
        self.source2i = {s: i for i, s in enumerate(self.i2source)}
        len_sources = len(self.i2source)

        self.additional_apps = ["APP_" + source for source in self.i2source if not self.additional_lexicon.contains("edge_labels", "APP_" + source)]
        self.additional_lexicon.sublexica["edge_labels"].add(self.additional_apps)
        len_labels = self.additional_lexicon.vocab_size("edge_labels")

        all_lex_types = {AMSentence.split_supertag(lextyp)[1] for lextyp, _ in self.additional_lexicon.sublexica["constants"] if "--TYPE--" in lextyp}
        self.i2lextyp = sorted(all_lex_types)
        self.lextyp2i : Dict[AMType, int] = { AMType.parse_str(l) : i for i, l in enumerate(self.i2lextyp)}
        len_lex_typ = len(self.i2lextyp)

        lexical2constant = np.zeros((len_lex_typ, self.additional_lexicon.vocab_size("constants")), dtype=np.bool) #shape (lexical type, constant)
        constant2lexical = np.zeros(self.additional_lexicon.vocab_size("constants"), dtype=np.long)

        get_term_types = np.zeros((len_lex_typ, len_labels, len_lex_typ), dtype=np.bool) #shape (parent lexical type, incoming label, term type)
        applyset_term_types = np.zeros((len_lex_typ, len_lex_typ, len_sources), dtype=np.bool) # shape (TERM TYPE, lexical type, source)
        apply_reachable_term_types = np.zeros((len_lex_typ, len_lex_typ), dtype=np.bool) # shape (TERM type, lexial type)

        #self.mod_cache = ModCache([AMType.parse_str(t) for t in all_lex_types])

        for constant,constant_id in self.additional_lexicon.sublexica["constants"]:
            lex_type = AMType.parse_str(AMSentence.split_supertag(constant)[1])
            lexical2constant[self.lextyp2i[lex_type], constant_id] = 1
            constant2lexical[constant_id] = self.lextyp2i[lex_type]

        apply_reachable_from : Dict[AMType, Set[Tuple[AMType, frozenset]]] = dict()
        for t1 in self.lextyp2i.keys():
            if t1.is_bot:
                continue
            for t2 in self.lextyp2i.keys():
                if t2.is_bot:
                    continue
                applyset = t1.get_apply_set(t2)
                if applyset is not None:
                    if t2 not in apply_reachable_from:
                        apply_reachable_from[t2] = set()
                    apply_reachable_from[t2].add((t1, frozenset(applyset)))

        root_id = self.additional_lexicon.get_id("edge_labels", "ROOT")
        for parent_lex_typ, parent_id in self.lextyp2i.items():
            # ROOT
            # root requires empty term type, thus all sources must be removed
            get_term_types[parent_id, root_id, parent_id] = True
            for current_lex_type in self.lextyp2i.keys():
                if current_lex_type.is_bot:
                    continue
                current_typ_id = self.lextyp2i[current_lex_type]
                apply_reachable_term_types[current_typ_id, current_typ_id] = True

            # MOD
            for source, t in self.mod_cache.get_modifiers(parent_lex_typ):
                smallest_apply_set : Dict[Tuple[int, str], Set[str]] = dict()
                if self.additional_lexicon.contains("edge_labels", "MOD_"+source):
                    label_id = self.additional_lexicon.get_id("edge_labels", "MOD_"+source)

                    get_term_types[parent_id, label_id, self.lextyp2i[t]] = True

                    for possible_lexical_type, applyset in apply_reachable_from[t]:
                        current_typ_id = self.lextyp2i[possible_lexical_type]

                        apply_reachable_term_types[self.lextyp2i[t], current_typ_id] = True
                        for source in applyset:
                            source_id = self.source2i[source]
                            applyset_term_types[self.lextyp2i[t], current_typ_id, source_id] = 1


            # APP
            for source in parent_lex_typ.nodes():
                req = parent_lex_typ.get_request(source)
                label_id = self.additional_lexicon.get_id("edge_labels", "APP_"+source)

                get_term_types[parent_id, label_id, self.lextyp2i[req]] = True

                for possible_lexical_type, applyset in apply_reachable_from[req]:
                    current_typ_id = self.lextyp2i[possible_lexical_type]

                    apply_reachable_term_types[self.lextyp2i[req], current_typ_id] = True
                    for source in applyset:
                        source_id = self.source2i[source]
                        applyset_term_types[self.lextyp2i[req], current_typ_id, source_id] = 1


        self.lexical2constant = torch.from_numpy(lexical2constant)
        self.constant2lexical = torch.from_numpy(constant2lexical)

        self.app_source2label_id = torch.zeros((len_sources, len_labels), dtype=torch.bool) # maps a source id to the respective (APP) label id
        self.mod_tensor = torch.zeros(len_labels, dtype=torch.bool) #which label ids are MOD_ edge labels?
        self.label_id2appsource = torch.zeros(len_labels, dtype=torch.long)-1
        self.applyset_term_types = torch.from_numpy(applyset_term_types) # shape (TERM TYPE, lexical type, source); is the given source in the apply set from the lexical type to the term type?
        self.get_term_types = torch.from_numpy(get_term_types) #shape (parent lexical type, incoming label, term type)
        self.apply_reachable_term_types = torch.from_numpy(apply_reachable_term_types) # shape (TERM type, lexial type)

        for label, label_id in self.additional_lexicon.sublexica["edge_labels"]:
            if label.startswith("MOD_"):
                self.mod_tensor[label_id] = True

        for source, source_id in self.source2i.items():
            label_id = self.additional_lexicon.get_id("edge_labels", "APP_"+source)
            self.label_id2appsource[label_id] = source_id
            self.app_source2label_id[source_id, label_id] = True

    def get_unconstrained_version(self) -> TransitionSystem:
        """
        Return an unconstrained version that does not do type checking.
        :return:
        """
        return GPUDFSChildrenFirst(self.children_order, self.pop_with_0, self.additional_lexicon, self.reverse_push_actions)

    def is_on_gpu(self):
        return True

    def prepare(self, device: Optional[int]):
        """
        Move precomputed arrays to GPU.
        :param device:
        :return:
        """
        self.lexical2constant = make_bool_multipliable(self.lexical2constant.to(device))
        self.constant2lexical = self.constant2lexical.to(device)

        self.app_source2label_id = make_bool_multipliable(self.app_source2label_id.to(device))
        self.mod_tensor = self.mod_tensor.to(device)
        self.label_id2appsource = self.label_id2appsource.to(device)

        self.applyset_term_types = make_bool_multipliable(self.applyset_term_types.to(device))
        self.get_term_types = self.get_term_types.to(device)
        self.apply_reachable_term_types = self.apply_reachable_term_types.to(device)

        self.apply_set_size = self.applyset_term_types.sum(dim=2) #shape (term types, lexical types)

    def guarantees_well_typedness(self) -> bool:
        return True

    def _add_missing_edge_scores(self, edge_scores : torch.Tensor) -> torch.Tensor:
        """
        Add edge scores for APP_x where x is a known source but APP_x was never seen in training
        :param edge_scores: shape (batch_size, number of edge_labels)
        :return:
        """
        if edge_scores.shape[1] == self.additional_lexicon.vocab_size("edge_labels"):
            return edge_scores
        additional_scores = torch.zeros((edge_scores.shape[0], len(self.additional_apps)), device=get_device_id(edge_scores))
        additional_scores -= 10_000_000 # unseen edges are very unlikely
        return torch.cat((edge_scores, additional_scores), dim=1)


    def gpu_initial_state(self, sentences : List[AMSentence], decoder_state : Any, device: Optional[int] = None) -> GPULTLState:
        max_len = max(len(s) for s in sentences)+1
        batch_size = len(sentences)
        stack = BatchedStack(batch_size, max_len+2, device=device)
        stack.push(torch.zeros(batch_size, dtype=torch.long, device=device), torch.ones(batch_size, dtype=torch.long, device=device))
        return GPULTLState(decoder_state, sentences, stack,
                           BatchedListofList(batch_size, max_len, max_len, device=device),
                           torch.zeros(batch_size, max_len, device=device, dtype=torch.long) - 1,  #heads
                           torch.zeros(batch_size, max_len, device=device, dtype=torch.long),  #labels
                           torch.zeros(batch_size, max_len, device=device, dtype=torch.long) - 1,  #constants
                           torch.zeros(batch_size, max_len, device=device, dtype=torch.long),  #lex labels
                           self.additional_lexicon,
                           torch.zeros(batch_size, max_len, device=device, dtype=torch.long) - 1,  #lexical types
                           make_bool_multipliable(torch.zeros((batch_size, max_len, len(self.i2source)), device=device, dtype=torch.bool)),  #apply set
                           )
        # decoder_state: Any
        # sentences : List[AMSentence]
        # stack: BatchedStack
        # children : BatchedListofList #shape (batch_size, input_seq_len, input_seq_len)
        # heads: torch.Tensor #shape (batch_size, input_seq_len) with 1-based id of parents, TO BE INITIALIZED WITH -1
        # edge_labels : torch.Tensor #shape (batch_size, input_seq_len) with the id of the incoming edge for each token
        # constants : torch.Tensor #shape (batch_size, input_seq_len)
        # lex_labels : torch.Tensor #shape (batch_size, input_seq_len)
        # lexicon : AdditionalLexicon

    def gpu_make_decision(self, scores: Dict[str, torch.Tensor], state : GPULTLState) -> DecisionBatch:
        children_scores = scores["children_scores"] #shape (batch_size, input_seq_len)
        batch_size, input_seq_len = children_scores.shape
        parent_mask = state.parent_mask()  #shape (batch_size, input_seq_len)
        mask = parent_mask #shape (batch_size, input_seq_len)
        active_nodes = state.stack.peek() #shape (batch_size,)
        batch_range = state.stack.batch_range #shape (batch_size,)
        applyset = state.applyset[batch_range, active_nodes] # shape (batch_size, sources);  applyset[b,s] = state.apply_set[b, active_nodes[b], s]
        done = state.stack.get_done() #shape (batch_size,)

        if state.step < 2:
            if state.step == 0: # nothing done yet, have to determine root
                # can only select a proper node when root not determined yet
                mask[:, 0] = 0
                push_mask = torch.ones(batch_size, dtype=torch.bool, device=get_device_id(children_scores))
                edge_labels = self.additional_lexicon.get_id("edge_labels","ROOT") + torch.zeros(batch_size, device=get_device_id(children_scores), dtype=torch.long)
            elif state.step == 1:
                # second step is always selecting 0 (pop artificial root)
                mask = torch.zeros_like(mask)
                mask[:, 0] = 1
                push_mask = torch.zeros(batch_size, dtype=torch.bool, device=get_device_id(children_scores))
                edge_labels = torch.argmax(scores["all_labels_scores"][:, 0], 1) #shape (batch_size,) -- dummy labels

            mask = (1-mask.long())*10_000_000
            _, selected_nodes = torch.max(children_scores - mask, dim=1)

            constants = torch.zeros_like(edge_labels)
            lex_labels = scores["lex_labels"]

            return DecisionBatch(selected_nodes, push_mask, ~push_mask, edge_labels, constants, None, lex_labels, ~push_mask)

        parents = state.heads[batch_range, active_nodes] #shape (batch_size,); parents[b] = state.heads[b,active_nodes[b]]
        lexical_type_parent = state.lex_types[batch_range, parents] #shape (batch_size, ); lexical_type_parent[b] = state.lex_types[b,parents[b]]
        assert lexical_type_parent.shape == (batch_size,)
        incoming_labels = state.edge_labels[batch_range, active_nodes] #shape (batch_size,) with ids of incoming edge labels of active nodes

        possible_term_types = self.get_term_types[lexical_type_parent, incoming_labels] #shape (batch_size, term types)
        assert possible_term_types.shape == (batch_size, len(self.lextyp2i))

        overlap = torch.einsum("tls, bs -> btl", self.applyset_term_types, applyset) #shape (batch_size, term_type, lexical)
        #overlap[b,t,l] = \sum_s self.apply_term_types[t,l,s] * applyset[b,s]

        #Now
        #mask out combinations that are not possible
        # either because the combination is not apply-reachable
        # or because the term type is not allowed here.
        overlap -= 10_000_000 * ~(self.apply_reachable_term_types.unsqueeze(0) * possible_term_types.unsqueeze(2))

        apply_set_size = applyset.sum(dim=1)
        consistent_term_type_lex_type_combos = are_eq(overlap, apply_set_size.unsqueeze(1).unsqueeze(2)) #shape (batch_size, term_type, lexical)
        # is the collected apply set a subset of the actual apply set? (and apply reachable)

        can_finish_now = are_eq(self.apply_set_size, overlap) #shape (batch_size, term_type, lexical)
        # containing the information whether the actual apply set is a subset of the collected apply set

        can_finish_now &= consistent_term_type_lex_type_combos  #shape (batch_size, term types, lexical types)
        can_finish_now = torch.any(can_finish_now, dim=1) #shape (batch_size, lexical types)

        assert can_finish_now.shape == (batch_size, len(self.lextyp2i))

        # we can potentially close the current node
        # if a) the stack is not empty already
        # and b) there is lexical type for our apply set and set of term types
        finishable = torch.any(can_finish_now, dim=1) #shape (batch_size,)
        assert finishable.shape == (batch_size, )
        depth = state.stack.depth() #shape (batch_size,)

        if self.pop_with_0:
            mask[batch_range, 0] &= (depth > 0)
            mask[:, 0] &= finishable
        else:
            mask[batch_range, active_nodes] &= (depth > 0)
            mask[batch_range, active_nodes] &= finishable

        mask = mask.long()
        mask *= state.position_mask()  # shape (batch_size, input_seq_len)

        mask = (1-mask)*10_000_000
        vals, selected_nodes = torch.max(children_scores - mask, dim=1)
        allowed_selection = vals > -1_000_000  # we selected something that was not extremely negative, shape (batch_size,)
        if self.pop_with_0:
            pop_mask = torch.eq(selected_nodes, 0)  #shape (batch_size,)
        else:
            pop_mask = torch.eq(selected_nodes, active_nodes)

        push_mask: torch.Tensor = (~pop_mask) & allowed_selection  # we push when we don't pop (but only if we are allowed to push)
        not_done = ~done
        push_mask &= not_done  # we can only push if we are not done with the sentence yet.
        pop_mask &= allowed_selection
        pop_mask &= not_done

        # compute constants for all instances (will only be used if pop_mask = True)
        # RE-USE the lexical types from above.
        possible_constants = index_or(make_bool_multipliable(can_finish_now), self.lexical2constant)
        assert possible_constants.shape == (batch_size, self.additional_lexicon.vocab_size("constants"))
        constant_mask = (~possible_constants).float()*10_000_000
        selected_constants = torch.argmax(scores["constants_scores"]-constant_mask, dim=1) #shape (batch_size,)

        # Edge labels
        # We create masks for what edges are appropriate

        # How many sources do we still need for a certain term type / lexical type combination?
        #                     (term type, lexical type)       (bs,)
        num_todo_sources = self.apply_set_size.unsqueeze(0) - apply_set_size.unsqueeze(1).unsqueeze(2) #shape (batch_size, term type, lexical type)
        # This uses the fact that |X - Y| = |X| - |Y| if Y is a subset of X, we catch the case ~(Y not subset of X) by considering only consistent combinations!

        # which combination of term type/lexical type does still work here (taking into account how many tokens are left)?
        consistent_term_type_lex_type_combos *= num_todo_sources <= state.w_c.unsqueeze(1).unsqueeze(2) #shape (batch_size, term type, lexical type)
        assert consistent_term_type_lex_type_combos.shape == (batch_size, len(self.lextyp2i), len(self.lextyp2i))

        num_todo_sources += 10_000_000 * (~consistent_term_type_lex_type_combos) #shape (batch_size, term type, lexical type)
        #exclude things that are not possible

        #possible_app_sources = torch.einsum("btl, tls -> bs", make_bool_multipliable(consistent_term_type_lex_type_combos),
        #                                    self.applyset_term_types) > 0
        possible_app_sources = torch.any(torch.any(consistent_term_type_lex_type_combos.unsqueeze(3) * self.applyset_term_types.unsqueeze(0).bool(), dim=1), dim=1)
        # shape (batch_size, source); possible_app_sourcwes[b,s] = \exists t,l such that consistent_term_type_lex_type_combos[b,t,l] AND self.applyset_term_types[t,l,s]

        #  mask out all sources that have been used already
        possible_app_sources &= ~applyset.bool()

        # MOD: all MOD_x edges are allowed provided that W_c - O_c >= 1
        o_c, _ = torch.min(torch.min(num_todo_sources,dim=1)[0], dim=1)
        assert o_c.shape == (batch_size, )

        if self.enable_assert:
            assert torch.all((o_c <= state.w_c) | done)

        #  we can use the fact that when the set of term types has the smallest apply set n and the largest apply set m, for all n <= i <= m, there is an apply set of size i.
        #o_c = torch.relu(minimal_apply_set_size - collected_apply_set_size)
        mod_mask = (state.w_c - o_c) >= 1 #shape (batch_size,)


        #  translate source mask to edge label mask
        edge_mask = index_or(make_bool_multipliable(possible_app_sources), self.app_source2label_id) #shape (batch_size, edge labels)

        edge_mask[mod_mask, :] |= self.mod_tensor #if we can use some MOD_x, we can use all MOD_x.
        edge_mask[:, self.additional_lexicon.get_id("edge_labels", "ROOT")] = False
        if self.enable_assert:
            assert torch.all(torch.any(edge_mask, dim=1) | pop_mask | done) #always at least one edge label (or we pop anyway).

        edge_scores = self._add_missing_edge_scores(scores["all_labels_scores"][state.stack.batch_range, selected_nodes]) #shape (batch_size, edge labels)

        edge_labels = torch.argmax(edge_scores - 10_000_000 * (~edge_mask).float(), 1)

        lex_labels = scores["lex_labels"]

        return DecisionBatch(selected_nodes, push_mask, pop_mask, edge_labels, selected_constants, None, lex_labels, pop_mask)




    def gpu_step(self, state: GPULTLState, decision_batch: DecisionBatch) -> None:
        """
        Applies a decision to a parsing state.
        :param state:
        :param decision_batch:
        :return:
        """
        next_active_nodes = state.stack.peek()
        state.children.append(next_active_nodes, decision_batch.push_tokens, decision_batch.push_mask)
        range_batch_size = state.stack.batch_range
        inverse_push_mask = (1-decision_batch.push_mask.long())

        state.w_c -= decision_batch.push_mask.int()
        state.heads[range_batch_size, decision_batch.push_tokens] = inverse_push_mask*state.heads[range_batch_size, decision_batch.push_tokens] + decision_batch.push_mask * next_active_nodes
        state.edge_labels[range_batch_size, decision_batch.push_tokens] = inverse_push_mask*state.edge_labels[range_batch_size, decision_batch.push_tokens] + decision_batch.push_mask * decision_batch.edge_labels

        #state.edge_labels_readable = [ [self.additional_lexicon.get_str_repr("edge_labels", e) for e in batch] for batch in state.edge_labels.numpy()]
        #Check if new edge labels are APP or MOD
        #if APP, add respective source to collected apply set.
        sources_used = self.label_id2appsource[decision_batch.edge_labels] #shape (batch_size,)
        app_mask = decision_batch.push_mask.bool() & (sources_used >= 0) #shape (batch_size,)
        float_apply_mask = make_bool_multipliable(app_mask) # int or bool values, depending on whether we are on CPU or GPU
        state.applyset[range_batch_size, next_active_nodes, sources_used] = ~app_mask * state.applyset[range_batch_size, next_active_nodes, sources_used] + float_apply_mask
        #a = state.applyset.numpy()
        # apply_set_for_debugging = []
        # for batch in state.applyset[range_batch_size, next_active_nodes].numpy():
        #     a = set()
        #     for i,x in enumerate(batch):
        #         if x:
        #             a.add(i)
        #     apply_set_for_debugging.append(a)

        lexical_types = self.constant2lexical[decision_batch.constants] #shape (batch_size,)
        inverse_constant_mask = (1-decision_batch.constant_mask.long())
        state.lex_types[range_batch_size, next_active_nodes] = inverse_constant_mask * state.lex_types[range_batch_size, next_active_nodes] + \
                                                                   decision_batch.constant_mask * lexical_types
        state.constants[range_batch_size, next_active_nodes] = inverse_constant_mask * state.constants[range_batch_size, next_active_nodes] + decision_batch.constant_mask * decision_batch.constants
        state.lex_labels[range_batch_size, next_active_nodes] = inverse_constant_mask*state.lex_labels[range_batch_size, next_active_nodes] + decision_batch.constant_mask * decision_batch.lex_labels

        pop_mask = decision_batch.pop_mask.bool() #shape (batch_size,)
        active_children = state.children.lol[range_batch_size, next_active_nodes] #shape (batch_size, max. number of children)
        push_all_children_mask = (active_children != 0) #shape (batch_size, max. number of children)
        push_all_children_mask &= pop_mask.unsqueeze(1) # only push those children where we will pop the current node from the top of the stack.

        state.stack.pop_and_push_multiple(active_children, pop_mask, push_all_children_mask, reverse=self.reverse_push_actions)

        state.step += 1

