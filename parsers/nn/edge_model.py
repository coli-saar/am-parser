from copy import deepcopy
from typing import Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Attention, FeedForward
from allennlp.nn import Activation
from allennlp.nn.util import masked_log_softmax

from topdown_parser.nn.biaffine import BiaffineAttention


class EdgeModel(Model):

    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)

    def set_input(self, encoded_input: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Set input for current batch
        :param mask: shape (batch_size, input_seq_len)
        :param encoded_input: shape (batch_size, input_seq_len, encoder output dim)
        :return:
        """
        raise NotImplementedError()

    def edge_scores(self, decoder: torch.Tensor) -> torch.Tensor:
        """
        Obtain edge existence scores
        :param decoder: shape (batch_size, decoder dim)
        :return: a tensor of shape (batch_size, input_seq_len) with log-probabilites, normalized over second dimension
        """
        raise NotImplementedError()


@EdgeModel.register("attention")
class AttentionEdgeModel(EdgeModel):
    """
    Wrapper for a simple attention edge model.
    """

    def __init__(self, vocab: Vocabulary, attention: Attention):
        super().__init__(vocab)
        self.attention = attention
        self.attention._normalize = False

        self.encoded_input: Optional[torch.Tensor] = None
        self.mask: Optional[torch.Tensor] = None

    def set_input(self, encoded_input: torch.Tensor, mask: torch.Tensor) -> None:
        self.encoded_input = encoded_input
        self.mask = mask

    def edge_scores(self, decoder: torch.Tensor) -> torch.Tensor:
        if self.encoded_input is None:
            raise ValueError("Please call set_input first")

        scores = self.attention(decoder, self.encoded_input, self.mask)  # (batch_size, input_seq_len)
        assert scores.shape == self.encoded_input.shape[:2]

        return scores #masked_log_softmax(scores, self.mask, dim=1)



@EdgeModel.register("mlp")
class MLPEdgeModel(EdgeModel):

    def __init__(self, vocab: Vocabulary, encoder_dim: int, hidden_dim: int, activation : Activation = torch.tanh):
        super().__init__(vocab)
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_dim
        self.activation = activation

        self.W = torch.nn.Linear(encoder_dim, hidden_dim)
        self.U = torch.nn.Linear(encoder_dim, hidden_dim)
        self.FinalLayer = torch.nn.Linear(hidden_dim, 1)

        self.input_before_concat = None
        self.mask = None

    def set_input(self, encoded_input: torch.Tensor, mask: torch.Tensor) -> None:
        self.input_before_concat = self.W(encoded_input) #(batch_size, input_seq_len, hidden_size)
        self.mask = mask # batch_size, input_seq_len

    def edge_scores(self, decoder: torch.Tensor) -> torch.Tensor:
        if self.input_before_concat is None:
            raise ValueError("Please call set_input first")

        decoder_before_concat = self.U(decoder) # (batch_size, hidden_size)
        concatentated = self.activation(decoder_before_concat.unsqueeze(1) + self.input_before_concat) # shape (batch_size, input_seq_len, hidden_size)
        before_softmax = self.FinalLayer(concatentated).squeeze(2) # (batch_size, input_seq_len)

        return before_softmax
        #return masked_log_softmax(before_softmax, self.mask, dim=1)



@EdgeModel.register("ma")
class MaEdgeModel(EdgeModel):

    def __init__(self, vocab: Vocabulary, mlp: FeedForward):
        super().__init__(vocab)
        self.head_mlp = mlp
        self.dep_mlp = deepcopy(mlp)

        self.biaffine_attention = BiaffineAttention(self.head_mlp.get_output_dim(), self.dep_mlp.get_output_dim())


    def set_input(self, encoded_input: torch.Tensor, mask: torch.Tensor) -> None:
        # encoded_input: (batch_size, seq_len, encoder_dim)
        self.batch_size, self.seq_len, _ = encoded_input.shape

        dependent_rep = self.dep_mlp(encoded_input) #(batch_size, seq_len, dependent_dim)
        self.biaffine_attention.set_input(dependent_rep)
        self.mask = mask

    def edge_scores(self, decoder: torch.Tensor) -> torch.Tensor:

        head_rep = self.head_mlp(decoder)
        raw_scores = self.biaffine_attention(head_rep) #shape (batch_size, seq_len)

        assert raw_scores.shape == (self.batch_size, self.seq_len)

        return raw_scores
        #return masked_log_softmax(raw_scores, self.mask, dim=1)
