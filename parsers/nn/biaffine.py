import math

import torch
from allennlp.modules import Attention
from allennlp.nn import Activation
from overrides import overrides
from torch.nn import Parameter, Module
import torch.nn.functional as F


#@Attention.register("biaffine")
class BiaffineAttention(Module):
    """
    Computes bi-affine attention, see Ma et al. 2018 section 3.5
    """

    def __init__(
            self,
            vector_dim: int,
            matrix_dim: int,
    ) -> None:
        super().__init__()
        self.vector_dim = vector_dim
        self.matrix_dim = matrix_dim

        self._weight_matrix = Parameter(torch.Tensor(vector_dim, matrix_dim))
        self._bias = Parameter(torch.Tensor(1))

        self.q_weight = Parameter(torch.Tensor(vector_dim))
        self.key_weight = Parameter(torch.Tensor(matrix_dim))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.vector_dim)
        torch.nn.init.uniform_(self.q_weight, -bound, bound)
        bound = 1 / math.sqrt(self.matrix_dim)
        torch.nn.init.uniform_(self.key_weight, -bound, bound)
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        torch.nn.init.constant_(self._bias, 0.)

    def set_input(self, matrix : torch.Tensor) -> None:
        """
        Set the input sentence.
        :param matrix:  shape (batch_size, num_tokens, matrix dim)
        :return:
        """
        self.intermediate_matrix = F.linear(matrix, self._weight_matrix) #shape (batch_size, num_tokens, vec dim)
        self.matrix_term = torch.einsum("brv, v -> br", matrix, self.key_weight) # (batch_size, num_tokens)

    @overrides
    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores to all tokens in the input sentence
        :param vector: shape (batch_size, vector dim)
        :return: shape (batch_size, num_tokens), where num_tokens is determined by call to set_input.
        """

        intermediate = torch.einsum("brv, bv -> br", self.intermediate_matrix, vector) #shape (batch_size, num_rows)

        intermediate = intermediate + self.matrix_term # shape (batch_size, num_rows)

        vector_term = torch.einsum("v, bv -> b", self.q_weight, vector) # shape (batch_size, )

        return intermediate + vector_term.unsqueeze(1) + self._bias #shape (batch_size, num_rows)