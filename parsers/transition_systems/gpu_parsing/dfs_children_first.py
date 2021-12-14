from typing import List, Iterable, Optional, Dict, Any

import torch

from parsers.am_algebra.tree import Tree
from parsers.dataset_readers.additional_lexicon import AdditionalLexicon
from parsers.dataset_readers.amconll_tools import AMSentence
from parsers.transition_systems.decision import DecisionBatch
from parsers.transition_systems.dfs_children_first import DFSChildrenFirst
from parsers.transition_systems.gpu_parsing.datastructures.list_of_list import BatchedListofList
from parsers.transition_systems.gpu_parsing.datastructures.stack import BatchedStack
from parsers.transition_systems.batched_parsing_state import BatchedParsingState
#from parsers.transition_systems.gpu_parsing.transition_system import GPUTransitionSystem, Decision, DecisionBatch
#from parsers.transition_systems.parsing_state import get_parent, get_siblings
#from parsers.transition_systems.unconstrained_system import UnconstrainedTransitionSystem
from parsers.transition_systems.transition_system import TransitionSystem


class GPUDFSChildrenFirstState(BatchedParsingState):
    pass

#@GPUTransitionSystem.register("dfs-children-first")
@TransitionSystem.register("dfs-children-first")
class GPUDFSChildrenFirst(DFSChildrenFirst):
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
        super().__init__(children_order, pop_with_0, additional_lexicon, reverse_push_actions)

    def is_on_gpu(self):
        """
        Do we parse on the GPU?
        :return:
        """
        return True

    def gpu_initial_state(self, sentences : List[AMSentence], decoder_state : Any, device: Optional[int] = None) -> GPUDFSChildrenFirstState:
        max_len = max(len(s) for s in sentences)+1
        batch_size = len(sentences)
        stack = BatchedStack(batch_size, max_len+2, device=device)
        stack.push(torch.zeros(batch_size, dtype=torch.long, device=device), torch.ones(batch_size, dtype=torch.long, device=device))
        return GPUDFSChildrenFirstState(decoder_state, sentences, stack,
                                        BatchedListofList(batch_size, max_len, max_len, device=device),
                                        torch.zeros(batch_size, max_len, device=device, dtype=torch.long) - 1,  #heads
                                        torch.zeros(batch_size, max_len, device=device, dtype=torch.long),  #labels
                                        torch.zeros(batch_size, max_len, device=device, dtype=torch.long) - 1,  #constants
                                        torch.zeros(batch_size, max_len, device=device, dtype=torch.long),  # term_types
                                        torch.zeros(batch_size, max_len, device=device, dtype=torch.long),  #lex labels
                                        self.additional_lexicon)
        # decoder_state: Any
        # sentences : List[AMSentence]
        # stack: BatchedStack
        # children : BatchedListofList #shape (batch_size, input_seq_len, input_seq_len)
        # heads: torch.Tensor #shape (batch_size, input_seq_len) with 1-based id of parents, TO BE INITIALIZED WITH -1
        # edge_labels : torch.Tensor #shape (batch_size, input_seq_len) with the id of the incoming edge for each token
        # constants : torch.Tensor #shape (batch_size, input_seq_len)
        # lex_labels : torch.Tensor #shape (batch_size, input_seq_len)
        # lexicon : AdditionalLexicon

    def gpu_make_decision(self, scores: Dict[str, torch.Tensor], state : BatchedParsingState) -> DecisionBatch:
        children_scores = scores["children_scores"] #shape (batch_size, input_seq_len)
        mask = state.parent_mask() #shape (batch_size, input_seq_len)
        depth = state.stack.depth() #shape (batch_size,)
        active_nodes = state.stack.peek()
        if self.pop_with_0:
            mask[state.stack.batch_range, 0] *= (depth > 0)
        else:
            mask[state.stack.batch_range, active_nodes] *= (depth > 0)

        mask = mask.long()
        mask *= state.position_mask()  # shape (batch_size, input_seq_len)

        mask = (1-mask)*10_000_000
        vals, selected_nodes = torch.max(children_scores - mask, dim=1)
        allowed_selection = vals > -1_000_000  # we selected something that was not extremely negative, shape (batch_size,)
        if self.pop_with_0:
            pop_mask = torch.eq(selected_nodes, 0)  #shape (batch_size,)
        else:
            pop_mask = torch.eq(selected_nodes, active_nodes)

        push_mask: torch.Tensor = (~pop_mask) * allowed_selection  # we push when we don't pop (but only if we are allowed to push)
        not_done = ~state.stack.get_done()
        push_mask *= not_done  # we can only push if we are not done with the sentence yet.
        pop_mask *= allowed_selection
        pop_mask *= not_done

        edge_labels = torch.argmax(scores["all_labels_scores"][state.stack.batch_range, selected_nodes], 1)
        constants = torch.argmax(scores["constants_scores"], 1)
        lex_labels = scores["lex_labels"]

        return DecisionBatch(selected_nodes, push_mask, pop_mask, edge_labels, constants, None, lex_labels, pop_mask)

    def gpu_step(self, state: GPUDFSChildrenFirstState, decision_batch: DecisionBatch) -> None:
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

        state.heads[range_batch_size, decision_batch.push_tokens] = inverse_push_mask*state.heads[range_batch_size, decision_batch.push_tokens] + decision_batch.push_mask * next_active_nodes
        state.edge_labels[range_batch_size, decision_batch.push_tokens] = inverse_push_mask*state.edge_labels[range_batch_size, decision_batch.push_tokens] + decision_batch.push_mask * decision_batch.edge_labels

        inverse_constant_mask = (1-decision_batch.constant_mask.long())
        state.constants[range_batch_size, next_active_nodes] = inverse_constant_mask * state.constants[range_batch_size, next_active_nodes] + decision_batch.constant_mask * decision_batch.constants
        state.lex_labels[range_batch_size, next_active_nodes] = inverse_constant_mask*state.lex_labels[range_batch_size, next_active_nodes] + decision_batch.constant_mask * decision_batch.lex_labels

        pop_mask = decision_batch.pop_mask.bool() #shape (batch_size,)
        active_children = state.children.lol[range_batch_size, next_active_nodes] #shape (batch_size, max. number of children)
        push_all_children_mask = (active_children != 0) #shape (batch_size, max. number of children)
        push_all_children_mask &= pop_mask.unsqueeze(1) # only push those children where we will pop the current node from the top of the stack.

        state.stack.pop_and_push_multiple(active_children, pop_mask, push_all_children_mask, reverse=self.reverse_push_actions)

