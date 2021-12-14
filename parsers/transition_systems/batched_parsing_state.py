from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from allennlp.nn.util import get_mask_from_sequence_lengths

from parsers.dataset_readers.amconll_tools import AMSentence
from parsers.dataset_readers.additional_lexicon import AdditionalLexicon
import torch.nn.functional as F

from parsers.transition_systems.gpu_parsing.datastructures.list_of_list import BatchedListofList
from parsers.transition_systems.gpu_parsing.datastructures.stack import BatchedStack
from parsers.nn.utils import get_device_id




@dataclass
class BatchedParsingState:

    decoder_state: Any
    sentences: List[AMSentence]
    stack: BatchedStack
    children: BatchedListofList #shape (batch_size, input_seq_len, input_seq_len)
    heads: torch.Tensor #shape (batch_size, input_seq_len) with 1-based id of parents, TO BE INITIALIZED WITH -1
    edge_labels: torch.Tensor #shape (batch_size, input_seq_len) with the id of the incoming edge for each token
    constants: torch.Tensor #shape (batch_size, input_seq_len), TO BE INITIALIZED WITH -1
    term_types: torch.Tensor #shape (batch_size, input_seq_len)
    lex_labels: torch.Tensor #shape (batch_size, input_seq_len)
    lexicon: AdditionalLexicon

    def parent_mask(self) -> torch.Tensor:
        """
        1 iff node still need head.
        :return: shape (batch_size, input_seq_len)
        """
        return self.heads < 0

    def get_lengths(self) -> torch.Tensor:
        if not hasattr(self, "actual_lengths"):
            self.actual_lengths = torch.tensor([len(s) for s in self.sentences], device=get_device_id(self.constants))
        return self.actual_lengths

    def position_mask(self) -> torch.Tensor:
        """
        Which elements are actual words in the sentence?
        :return: shape (batch_size, input_seq_len)
        """
        if not hasattr(self, "lengths"):
            self.lengths = torch.tensor([len(s)+1 for s in self.sentences], device=get_device_id(self.constants))
            self.max_len = max(len(s) for s in self.sentences)+1

        return get_mask_from_sequence_lengths(self.lengths, self.max_len)

    def constant_mask(self) -> torch.Tensor:
        """
        1 iff node doesn't have constant chosen yet.
        :return:
        """
        return self.constants < 0

    def is_complete(self) -> torch.Tensor:
        """
        All sentences in the batch done?
        :return:
        """
        return torch.all(self.stack.is_empty())

    def extract_trees(self) -> List[AMSentence]:
        r = []
        heads = F.relu(self.heads[:, 1:]).cpu().numpy()
        label_tensor = self.edge_labels[:, 1:].cpu().numpy()
        constant_tensor = self.constants[:, 1:].cpu().numpy()
        lex_label_tensor = self.lex_labels[:, 1:].cpu().numpy()
        for i in range(len(self.sentences)):
            length = len(self.sentences[i])
            sentence = self.sentences[i].set_heads([max(0, h) for i, h in enumerate(heads[i]) if i < length]) #negative heads are ignored, so point to 0.
            edge_labels = [self.lexicon.get_str_repr("edge_labels", id) if id > 0 else "IGNORE" for i, id in enumerate(label_tensor[i]) if i < length]
            sentence = sentence.set_labels(edge_labels)
            constants = [AMSentence.split_supertag(self.lexicon.get_str_repr("constants", id)) if id > 0 else AMSentence.split_supertag(AMSentence.get_bottom_supertag())
                         for i, id in enumerate(constant_tensor[i]) if i < length]
            sentence = sentence.set_supertag_tuples(constants)
            lex_labels = [self.lexicon.get_str_repr("lex_labels", id) if id > 0 else "_" for i, id in enumerate(lex_label_tensor[i]) if i < length]
            sentence = sentence.set_lexlabels(lex_labels)
            r.append(sentence)
        return r

    def gather_context(self) -> Dict[str, torch.Tensor]:
        """
        Extracts features of the current context like siblings, grandparents etc.
        :return: a dictionary which can be used for storing
        various kinds of information, we can condition on.
        """
        active_nodes = self.stack.peek() #shape (batch_size,)
        context = {"parents": F.relu(self.heads[self.stack.batch_range, active_nodes]), "children": self.children.outer_index(active_nodes),
                   "children_mask" : self.children.outer_index(active_nodes) != 0}
        return context

    def copy(self) -> "BatchedParsingState":
        """
        A way of copying this parsing state such that modifying objects that constrain the future
        will be modifying copied objects. e.g. we need a deep copy of the stack and nodes seen already
        but we don't need a deep copy of the decoder state or the lexicon.
        :return:
        """
        raise NotImplementedError()


