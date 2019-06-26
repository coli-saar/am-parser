from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask
import torch
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward
import copy
from allennlp.nn import Activation
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import InputVariationalDropout
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from typing import Dict, Optional, Tuple, Any, List
from allennlp.models.model import Model

import torch.nn.functional as F
from allennlp.nn.chu_liu_edmonds import decode_mst
import numpy

from graph_dependency_parser.components.edge_models.base import EdgeModel


@EdgeModel.register("dm_edges")
class DMEdges(EdgeModel):
    """
    This dependency edge model follows the model of
    ` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
    <https://arxiv.org/abs/1611.01734>`_ .
    """

    def __init__(self, vocab: Vocabulary,
                 encoder_dim: int,
                 label_dim: int,
                 edge_dim: int,
                 edge_label_namespace: str,
                 dropout: float,
                 tag_feedforward: FeedForward = None,
                 arc_feedforward: FeedForward = None) -> None:
        """
        Parameters
        ----------
        vocab : ``Vocabulary``, required
            A Vocabulary, required in order to compute sizes for input/output projections.
        encoder_dim : ``int``, required.
            The output dimension of the encoder.
        label_dim : ``int``, required.
            The dimension of the MLPs used for dependency tag prediction.
        edge_dim : ``int``, required.
            The dimension of the MLPs used for head arc prediction.
        edge_label_namespace: str,
                The namespace of the edge labels: a combination of the task name + _head_tags
        tag_feedforward : ``FeedForward``, optional, (default = None).
            The feedforward network used to produce tag representations.
            By default, a 1 layer feedforward network with an elu activation is used.
        arc_feedforward : ``FeedForward``, optional, (default = None).
            The feedforward network used to produce arc representations.
            By default, a 1 layer feedforward network with an elu activation is used.
        dropout : ``float``, optional, (default = 0.0)
            The variational dropout applied to the output of the encoder and MLP layers.
        """
        super(DMEdges, self).__init__(vocab)
        self._encoder_dim = encoder_dim

        self.head_arc_feedforward = arc_feedforward or \
                                    FeedForward(encoder_dim, 1,
                                                edge_dim,
                                                Activation.by_name("elu")())
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(edge_dim,
                                                     edge_dim,
                                                     use_input_biases=True)

        num_labels = vocab.get_vocab_size(edge_label_namespace)

        self.head_tag_feedforward = tag_feedforward or \
                                    FeedForward(encoder_dim, 1,
                                                label_dim,
                                                Activation.by_name("elu")())
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(label_dim,
                                                      label_dim,
                                                      num_labels)

        self._dropout = InputVariationalDropout(dropout)

        check_dimensions_match(label_dim, self.head_tag_feedforward.get_output_dim(),
                               "tag representation dim", "tag feedforward output dim")
        check_dimensions_match(edge_dim, self.head_arc_feedforward.get_output_dim(),
                               "arc representation dim", "arc feedforward output dim")

    def encoder_dim(self):
        return self._encoder_dim

    def edge_existence(self, encoded_text: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes edge existence scores for a batch of sentences.

        Parameters
        ----------
        encoded_text : torch.Tensor, required
            The input sentence, with artificial root node (head sentinel) added in the beginning of
            shape (batch_size, sequence length, encoding dim)

        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.

        Returns
        -------
        attended_arcs: torch.Tensor
            The edge existence scores in a tensor of shape (batch_size, sequence_length, sequence_length). The mask is taken into account.
        """
        float_mask = mask.float()

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation,
                                           child_arc_representation)

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        return attended_arcs

    def label_scores(self, encoded_text: torch.Tensor, head_indices: torch.Tensor) -> torch.Tensor:
        """
        Computes edge label scores for a fixed tree structure (given by head_indices) for a batch of sentences.

        Parameters
        ----------
        encoded_text: (batch_size, sequence_length, encoder_output_dim)

        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word (predicted or gold).

        Returns
        -------
        head_tag_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(
            encoded_text))  # will be used to generate predictions for the edge labels for the given arcs.
        child_tag_representation = self._dropout(self.child_tag_feedforward(
            encoded_text))  # will be used to generate predictions for the edge labels for the given arcs.

        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, get_device_of(head_tag_representation)).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag_representations,
                                            child_tag_representation)
        return head_tag_logits

    def full_label_scores(self, encoded_text:torch.Tensor) -> torch.Tensor:
        """
        Computes edge label scores for all edges for a batch of sentences.

        Parameters
        ----------
         encoded_text : torch.Tensor, required
            The input sentence, with artifical root node (head sentinel) added in the beginning of
            shape (batch_size, sequence length, encoding dim)

        Returns
        -------
        edge_label_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length,sequence_length, num_edge_labels),
            representing logits for predicting a distribution over edge labels
            for each edge. [i,j,k,l] is the the score for edge j->k being labeled l in sentence i
        """
        raise NotImplementedError("full_label_scores is not yet implemented for the Dozat&Manning edge model")
