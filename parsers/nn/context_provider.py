from typing import Dict, Any, List

import torch
from allennlp.common import Registrable
from allennlp.modules import FeedForward
from allennlp.nn.util import get_lengths_from_binary_sequence_mask

from torch.nn import Module, EmbeddingBag, Dropout, Embedding

from parsers.dataset_readers.additional_lexicon import AdditionalLexicon
from parsers.nn.utils import get_device_id


class ContextProvider(Module, Registrable):
    """
    Takes the context extracted by gather_context() in the transition system
    and computes a fixed vector that is used as input to the decoder.
    """

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the representation that is used as input to the decoder network.
        :param current_node: tensor of shape (batch_size, encoder dim) with representation of active nodes.
        :param state: contains keys "encoded_input", which contains the entire input in a tensor of shape (batch_size, input_seq_len, encoder dim)
        :param context: provided by gather_context a dictionary with values of shape (batch_size, *) with additional dimensions for the current time step.
        :return: of shape (batch_size, decoder_dim)
        """
        raise NotImplementedError()

    def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Takes the context and distills it into a vector that can then be combined in forward with the representation of the current node.
        :param state:
        :param context:
        :return:
        """
        raise NotImplementedError()

    def conditions_on(self) -> List[str]:
        """
        Returns the dictionary keys that it conditions on. Useful to know when doing beam search.
        :return:
        """
        raise NotImplementedError()


@ContextProvider.register("no_context")
class NoContextProvider(ContextProvider):

    def compute_context(self, state: Dict[str, torch.Tensor], context: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, Any]) -> torch.Tensor:
        return current_node

    def conditions_on(self) -> List[str]:
        return []

@ContextProvider.register("parent")
class ParentContextProvider(ContextProvider):
    """
    Add parent information in Ma et al. 2018 style
    """

    def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = state["encoded_input"].shape[0]

        parents = context["parents"] # shape (batch_size,)

        encoded_parents = state["encoded_input"][range(batch_size), parents] # shape (batch_size, encoder dim)

        #mask = parents == 0 # which parents are 0? Skip those
        #encoded_parents=  mask.unsqueeze(1) * encoded_parents

        return encoded_parents

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, Any]) -> torch.Tensor:

        return current_node + self.compute_context(state, context)

    def conditions_on(self) -> List[str]:
        return ["parents"]


class MostRecent(ContextProvider):
    """
    Add information about most recent sibling/child.
    """

    def __init__(self, context_key : str, mask_key : str):
        super().__init__()
        self.context_key = context_key
        self.mask_key = mask_key

    def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        # For the sake of the example, let's say we're looking for siblings
        siblings = context[self.context_key] #shape (batch_size, max_num_siblings)
        batch_size, _ = siblings.shape

        sibling_mask = context[self.mask_key] # (batch_size, max_num_siblings)

        number_of_siblings = get_lengths_from_binary_sequence_mask(sibling_mask) # (batch_size,)

        most_recent_sibling = siblings[range(batch_size), number_of_siblings-1] # shape (batch_size,)

        encoded_sibling = state["encoded_input"][range(batch_size), most_recent_sibling] # shape (batch_size, encoder_dim)

        # Some nodes don't have siblings, mask them out:
        encoded_sibling = (number_of_siblings != 0).unsqueeze(1) * encoded_sibling #shape (batch_size, encoder_dim)
        return encoded_sibling

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:

        return current_node + self.compute_context(state, context)


@ContextProvider.register("most-recent-child")
class SiblingContextProvider(ContextProvider):
    """
    Add information about most recent sibling, like Ma et al.
    """

    def __init__(self):
        super().__init__()
        self.most_recent = MostRecent("children", "children_mask")

    def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.most_recent.compute_context(state, context)

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:

        return current_node + self.compute_context(state, context)

    def conditions_on(self) -> List[str]:
        return ["children"]


@ContextProvider.register("sum")
class SumContextProver(ContextProvider):
    """
    Add information about most recent sibling, like Ma et al.
    """

    def __init__(self, providers : List[ContextProvider]):
        super().__init__()
        self.providers = providers
        for i, p in enumerate(providers):
            self.add_module("_sum_context_provider_"+str(i), p)

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        r = current_node

        for provider in self.providers:
            r = r + provider.compute_context(state, context)

        return r

    def conditions_on(self) -> List[str]:
        r = []
        for p in self.providers:
            r.extend(p.conditions_on())
        return r


@ContextProvider.register("plain-concat")
class PlainConcatContextProver(ContextProvider):
    """
    Concatenate additional information together.
    """

    def __init__(self, providers : List[ContextProvider]):
        super().__init__()
        self.providers = providers
        for i, p in enumerate(providers):
            self.add_module("_plain_concat_context_provider_"+str(i), p)

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        contexts = [current_node] #shape (batch_size, some dimension)

        for provider in self.providers:
            contexts.append(provider.compute_context(state, context))

        return torch.cat(contexts, dim=1)

    def conditions_on(self) -> List[str]:
        r = []
        for p in self.providers:
            r.extend(p.conditions_on())
        return r


# @ContextProvider.register("most-recent-sibling")
# class SiblingContextProvider(ContextProvider):
#     """
#     Add information about most recent sibling.
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.most_recent = MostRecent("siblings", "siblings_mask")
#
#     def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#         return self.most_recent.compute_context(state, context)
#
#     def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#
#         return current_node + self.compute_context(state, context)
#
#     def conditions_on(self) -> List[str]:
#         return ["siblings"]
#

# def _select(encoded_input : torch.Tensor, siblings : torch.Tensor) -> torch.Tensor:
#     """
#     Returns a tensor of shape (batch_size, max_num_siblings, encoder dim).
#
#     :param encoded_input: shape (batch_size, input_seq_len, encoder dim)
#     :param siblings: shape (batch_size, max_num_siblings)
#     :return:
#     """
#     batch_size, input_seq_len, encoder_dim = encoded_input.shape
#     _, max_num_siblings = siblings.shape
#
#     r = torch.zeros((batch_size, max_num_siblings, encoder_dim), device=get_device_id(encoded_input))
#
#     for b in range(batch_size):
#         r[b] = encoded_input[b,siblings[b]]
#
#     return r
#
# class AllNeighbors(ContextProvider):
#     """
#     Add information about all siblings/children.
#     """
#
#     def __init__(self, context_key : str, mask_key : str):
#         super().__init__()
#         self.context_key = context_key
#         self.mask_key = mask_key
#
#
#     def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#         # For the sake of the example, let's say we're looking for siblings
#         siblings = context[self.context_key] #shape (batch_size, max_num_siblings)
#         batch_size, max_num_siblings = siblings.shape
#
#         sibling_mask = context[self.mask_key] # (batch_size, max_num_siblings)
#
#         selected_nodes = _select(state["encoded_input"], siblings) #shape (batch_size, max_num_siblings, encoder_dim)
#
#         return torch.sum(selected_nodes * sibling_mask.unsqueeze(2), dim=1)
#
#     def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#
#         return current_node + self.compute_context(state, context)
#
# @ContextProvider.register("all-children")
# class AllChildrenContextProvider(ContextProvider):
#     """
#     Add information about most recent sibling, like Ma et al.
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.all_neighbors = AllNeighbors("children", "children_mask")
#
#     def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#         return self.all_neighbors.compute_context(state, context)
#
#     def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#
#         return current_node + self.compute_context(state, context)
#
#     def conditions_on(self) -> List[str]:
#         return ["children"]
#


# @ContextProvider.register("concat")
# class ConcatContextProver(ContextProvider):
#     """
#     Concatenate additional information together.
#     """
#
#     def __init__(self, providers : List[ContextProvider], mlp : FeedForward):
#         super().__init__()
#         self.mlp = mlp
#         self.providers = providers
#         for i, p in enumerate(providers):
#             self.add_module("_concat_context_provider_"+str(i), p)
#
    # def set_batch_range(self, batch_range: torch.Tensor) -> None:
    #     self.batch_range = batch_range
    #     for p in self.providers:
    #         p.set_batch_range(batch_range)
#     def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#         contexts = [current_node] #shape (batch_size, some dimension)
#
#         for provider in self.providers:
#             contexts.append(provider.compute_context(state, context))
#
#         return self.mlp(torch.cat(contexts, dim=1))
#
#     def conditions_on(self) -> List[str]:
#         r = []
#         for p in self.providers:
#             r.extend(p.conditions_on())
#         return r
#
# @ContextProvider.register("label-embedder")
# class LabelContextProvider(ContextProvider):
#     """
#     Add information about labels of other children.
#     """
#
#     def __init__(self, additional_lexicon : AdditionalLexicon, hidden_dim : int, dropout : float = 0.0):
#         super().__init__()
#         self.additional_lexicon = additional_lexicon
#         self.embeddings = EmbeddingBag(additional_lexicon.sublexica["edge_labels"].vocab_size(), hidden_dim, mode="sum")
#         self.dropout = Dropout(dropout)
#
#     def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#         edge_labels = context["children_labels"] #shape (batch_size, max number of children)
#
#         children_mask = context["children_mask"] # (batch_size, max number of children)
#
#         return self.dropout(self.embeddings(edge_labels, per_sample_weights = children_mask.float()))
#
#     def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#         return current_node + self.compute_context(state, context)
#
#     def conditions_on(self) -> List[str]:
#         return ["children_labels"]
#
# @ContextProvider.register("last-label-embedder")
# class LastLabelEmbedder(ContextProvider):
#     """
#     Add information about most recent label of child.
#     """
#
#     def __init__(self, additional_lexicon : AdditionalLexicon, hidden_dim : int):
#         super().__init__()
#         self.additional_lexicon = additional_lexicon
#         self.embeddings = Embedding(additional_lexicon.sublexica["edge_labels"].vocab_size(), hidden_dim)
#
#     def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#         edge_labels = context["children_labels"] #shape (batch_size, max number of children)
#
#         children_mask = context["children_mask"] # (batch_size, max number of children)
#         batch_size, _ = children_mask.shape
#
#         number_of_children = get_lengths_from_binary_sequence_mask(children_mask) # (batch_size,)
#
#         most_recent_label = edge_labels[self.batch_range, number_of_children-1] # shape (batch_size,)
#
#         encoded_label = self.embeddings(most_recent_label) # shape (batch_size, embedding dim)
#
#         # Some nodes don't have children, mask them out:
#         encoded_label = (number_of_children != 0).unsqueeze(1) * encoded_label #shape (batch_size, embedding dim)
#         return encoded_label
#
#     def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#
#         return current_node + self.compute_context(state, context)
#
#     def conditions_on(self) -> List[str]:
#         return ["children_labels"]
#
# @ContextProvider.register("type-embedder")
# class TypeContextProvider(ContextProvider):
#     """
#     Add information about lexical type of current node.
#     """
#
#     def __init__(self, additional_lexicon : AdditionalLexicon, hidden_dim : int, dropout : float = 0.0):
#         super().__init__()
#         self.additional_lexicon = additional_lexicon
#         self.embeddings = Embedding(additional_lexicon.sublexica["term_types"].vocab_size(), hidden_dim) #term_types is the name but it really is all types!
#         self.dropout = Dropout(dropout)
#
#     def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#         supertags = context["lexical_types"] #shape (batch_size, )
#
#         return self.dropout(self.embeddings(supertags))
#
#     def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
#         return current_node + self.compute_context(state, context)
#
#     def conditions_on(self) -> List[str]:
#         return ["lexical_types"]