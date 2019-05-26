from allennlp.nn.util import get_range_vector
from allennlp.nn.util import get_device_of
import torch
from allennlp.data import Vocabulary
from allennlp.modules import InputVariationalDropout
from allennlp.nn.activations import Activation
import numpy as np

from graph_dependency_parser.components.edge_models.base import EdgeModel


@EdgeModel.register("kg_edges")
class KGEdges(EdgeModel):
    """
    Reimplementation of the edge model of the graph-based graph_dependency_parser by Kiperwasser and Goldberg (2016): https://aclweb.org/anthology/Q16-1023
    """
    def __init__(self, vocab:Vocabulary,
                 encoder_dim:int,
                 label_dim:int,
                 edge_dim:int,
                 dropout: float,
                 edge_label_namespace: str,
                 activation : Activation = None) -> None:
        """
            Parameters
            ----------
            vocab : ``Vocabulary``, required
                A Vocabulary, required in order to compute sizes for input/output projections.
            encoder_dim : ``int``, required.
                The output dimension of the encoder.
            label_dim : ``int``, required.
                The dimension of the hidden layer of the MLP used for predicting the edge labels.
            edge_dim : ``int``, required.
                The dimension of the hidden layer of the MLP used for predicting edge existence.
            edge_label_namespace: str,
                The namespace of the edge labels: a combination of the task name + _head_tags
            activation : ``Activation``, optional, (default = tanh).
                The activation function used in the MLPs.
            dropout : ``float``, optional, (default = 0.0)
                The variational dropout applied to the output of the encoder and MLP layers.
        """
        super(KGEdges, self).__init__(vocab)
        self._encoder_dim = encoder_dim
        if activation is None:
            self.activation = Activation.by_name("tanh")()
        else:
            self.activation = activation

        #edge existence:

        #these two matrices together form the feed forward network which takes the vectors of the two words in question and makes predictions from that
        #this is the trick described by Kiperwasser and Goldberg to make training faster.
        self.head_arc_feedforward = torch.nn.Linear(encoder_dim, edge_dim)
        self.child_arc_feedforward = torch.nn.Linear(encoder_dim, edge_dim, bias=False) #bias is already added by head_arc_feedforward

        self.arc_out_layer = torch.nn.Linear(edge_dim, 1, bias=False)  # K&G don't use a bias for the output layer

        #edge labels:
        num_labels = vocab.get_vocab_size(edge_label_namespace)

        #same trick again
        self.head_label_feedforward = torch.nn.Linear(encoder_dim, label_dim)
        self.child_label_feedforward = torch.nn.Linear(encoder_dim, label_dim, bias=False)

        self.label_out_layer = torch.nn.Linear(edge_dim, num_labels) #output layer for edge labels

        self._dropout = InputVariationalDropout(dropout)

    def encoder_dim(self):
        return self._encoder_dim

    def edge_existence(self,encoded_text: torch.Tensor, mask : torch.LongTensor) -> torch.Tensor:
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

        bs,sl,arc_dim = head_arc_representation.size()

        #now repeat the token representations to form a matrix:
        #shape (batch_size, sequence_length, sequence_length, arc_representation_dim)
        heads = head_arc_representation.repeat(1,sl,1).reshape(bs,sl,sl,arc_dim) #heads in one direction
        deps = child_arc_representation.repeat(1, sl, 1).reshape(bs, sl, sl, arc_dim).transpose(1,2) #deps in the other direction

        # shape (batch_size, sequence_length, sequence_length, arc_representation_dim)
        combined = self.activation(heads + deps) #now the feedforward layer that takes every pair of vectors for tokens is complete.
        #combined now represents the activations in the hidden layer of the MLP.
        edge_scores = self.arc_out_layer(combined).squeeze(3) #last dimension is now 1, remove it

        #mask out stuff for padded tokens:
        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        edge_scores = edge_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        return edge_scores

    def label_scores(self, encoded_text:torch.Tensor, head_indices: torch.Tensor) -> torch.Tensor:
        """
        Computes edge label scores for a fixed tree structure (given by head_indices) for a batch of sentences.

        Parameters
        ----------
         encoded_text : torch.Tensor, required
            The input sentence, with artifical root node (head sentinel) added in the beginning of
            shape (batch_size, sequence length, encoding dim)
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word (predicted or gold).

        Returns
        -------
        edge_label_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each given arc.
        """
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_label_representation = self._dropout(self.head_label_feedforward(encoded_text))
        child_label_representation = self._dropout(self.child_label_feedforward(encoded_text))

        batch_size = head_label_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, get_device_of(head_label_representation)).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_label_representations = head_label_representation[range_vector, head_indices]
        selected_head_label_representations = selected_head_label_representations.contiguous()

        combined = self.activation(selected_head_label_representations + child_label_representation)
        #(batch_size, sequence_length, num_head_tags)
        edge_label_logits = self.label_out_layer(combined)

        return edge_label_logits
