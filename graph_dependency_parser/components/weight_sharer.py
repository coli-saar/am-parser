from typing import List, Tuple, Union

import torch
from allennlp.common import Registrable
from allennlp.modules import Seq2SeqEncoder
from torch.nn import Module, Dropout
import copy


class MTLWeightSharer(Registrable, Module):
    """
    A module that creates a task specific representation for each word of a sentence.
    It returns a tuple of two tensors:
    - the first tensor is a (batch_size, seq_len, encoder_output_dim) tensor that will be used as input for the parsing model
    - the second is a tensor of same shape to be used as input for the tagging model, or None
    The second tensor can be None only if the passed formalism is one of those where tagging is not required
    """
    def __init__(self, encoder: Seq2SeqEncoder, formalisms: List[str], formalisms_without_tagging: List[str], task_dropout:float = 0.0):
        """

        :param encoder: the encoder described in the model configuration, potentially the bases for the weight sharer
        :param formalisms: the names of the formalisms we are tackling.
        :param formalisms_without_tagging: the names of formalisms for which we don't need a tagging representation
        :param task_dropout, dropout to apply to task-specific encoders (but not to shared ones), if applicable
        """
        super().__init__()

    def forward(self, formalism: str, word_rep: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Takes a formalism, the representation of the words and a mask and produces two (potentially the same) tensor,
        one that is used for input to the edge models, the second one is used as input to the tagging models
        :param formalism:
        :param word_rep:
        :param mask:
        :return: encoded_text_for_parsing, encoded_text_for_tagging, shape (batch_size, seq_len, encoder_dim)
        """
        raise NotImplementedError()

    def get_input_dim(self):
        raise NotImplementedError()

    def get_output_dim(self):
        raise NotImplementedError()



@MTLWeightSharer.register("shared_encoder")
class SharedEncoder(MTLWeightSharer):
    """
    A shared encoder over all tasks, uses the same tensor as input to edge models and supertagging models.
    """
    def __init__(self, encoder: Seq2SeqEncoder, formalisms: List[str], formalisms_without_tagging: List[str], task_dropout:float = 0.0):
        super().__init__(encoder, formalisms, formalisms_without_tagging, task_dropout)
        self.encoder = encoder

    def forward(self, formalism: str, word_rep: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Ignores the formalism and computes a shared representation.
        :param formalism:
        :param word_rep:
        :param mask:
        :return:
        """
        encoded_text : torch.Tensor = self.encoder(word_rep,mask)
        return encoded_text, encoded_text

    def get_input_dim(self):
        return self.encoder.get_input_dim()

    def get_output_dim(self):
        return self.encoder.get_output_dim()



@MTLWeightSharer.register("shared_split_encoder")
class SharedDifferentEncoder(MTLWeightSharer):
    """
    A shared encoder over all tasks, but separate encoders for input to edge model and to supertagging model.
    """
    def __init__(self, encoder: Seq2SeqEncoder, formalisms: List[str], formalisms_without_tagging: List[str], task_dropout:float = 0.0):
        super().__init__(encoder, formalisms, formalisms_without_tagging, task_dropout)
        self.encoder_parsing = encoder
        self.encoder_tagging = copy.deepcopy(encoder)

    def forward(self, formalism: str, word_rep: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Ignores the formalism and computes a shared representation.
        :param formalism:
        :param word_rep:
        :param mask:
        :return:
        """
        encoded_text_parsing : torch.Tensor = self.encoder_parsing(word_rep,mask)
        encoded_text_tagging : torch.Tensor = self.encoder_tagging(word_rep,mask)
        return encoded_text_parsing, encoded_text_tagging

    def get_input_dim(self):
        return self.encoder_parsing.get_input_dim()

    def get_output_dim(self):
        return self.encoder_parsing.get_output_dim()


@MTLWeightSharer.register("freda")
class Freda(MTLWeightSharer):

    def __init__(self, encoder: Seq2SeqEncoder, formalisms: List[str], formalisms_without_tagging: List[str], task_dropout:float = 0.0):
        super().__init__(encoder, formalisms, formalisms_without_tagging, task_dropout)
        self.shared_encoder = encoder
        self.task_dropout = Dropout(task_dropout)
        self.task_specific_encoders = { formalism: copy.deepcopy(encoder) for formalism in formalisms }
        for formalism,encoder in self.task_specific_encoders.items():
            self.add_module(formalism,encoder)

    def forward(self, formalism: str, word_rep: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """

        :param formalism:
        :param word_rep:
        :param mask:
        :return:
        """
        shared = self.shared_encoder(word_rep, mask) #shape (batch_size, seq_len, encoder_dim)
        formalism_specific = self.task_dropout(self.task_specific_encoders[formalism](word_rep, mask)) #shape (batch_size, seq_len, encoder_dim)
        combined = torch.cat([shared,formalism_specific],dim=2) # #shape (batch_size, seq_len, 2* encoder_dim)
        return combined, combined

    def get_input_dim(self):
        return self.shared_encoder.get_input_dim()

    def get_output_dim(self):
        return self.shared_encoder.get_output_dim() * 2


@MTLWeightSharer.register("freda_split")
class FredaSplit(MTLWeightSharer):

    def __init__(self, encoder: Seq2SeqEncoder, formalisms: List[str], formalisms_without_tagging: List[str], task_dropout:float = 0.0):
        super().__init__(encoder, formalisms, formalisms_without_tagging)
        self.shared_encoder_parsing = encoder
        self.shared_encoder_tagging = copy.deepcopy(encoder)
        self.task_dropout = Dropout(task_dropout)
        self.task_specific_encoders_parsing = { formalism: copy.deepcopy(encoder) for formalism in formalisms }
        self.task_specific_encoders_tagging = { formalism: copy.deepcopy(encoder) for formalism in set(formalisms) - set(formalisms_without_tagging) }
        for formalism,encoder in self.task_specific_encoders_parsing.items():
            self.add_module(formalism+"_parsing",encoder)
        for formalism,encoder in self.task_specific_encoders_tagging.items():
            self.add_module(formalism+"_tagging",encoder)

    def forward(self, formalism: str, word_rep: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """

        :param formalism:
        :param word_rep:
        :param mask:
        :return:
        """
        shared_parsing = self.shared_encoder_parsing(word_rep, mask) #shape (batch_size, seq_len, encoder_dim)
        shared_tagging = self.shared_encoder_tagging(word_rep, mask) #shape (batch_size, seq_len, encoder_dim)

        formalism_specific_parsing = self.task_dropout(self.task_specific_encoders_parsing[formalism](word_rep, mask)) #shape (batch_size, seq_len, encoder_dim)
        combined_parsing = torch.cat([shared_parsing,formalism_specific_parsing],dim=2) #shape (batch_size, seq_len, 2* encoder_dim)

        if formalism in self.task_specific_encoders_tagging:
            formalism_specific_tagging = self.task_dropout(self.task_specific_encoders_tagging[formalism](word_rep,mask))  # shape (batch_size, seq_len, encoder_dim)
            combined_tagging = torch.cat([shared_tagging,formalism_specific_tagging],dim=2) #shape (batch_size, seq_len, 2* encoder_dim)
        else:
            combined_tagging = None

        return combined_parsing, combined_tagging

    def get_input_dim(self):
        return self.shared_encoder_parsing.get_input_dim()

    def get_output_dim(self):
        return self.shared_encoder_tagging.get_output_dim() * 2
