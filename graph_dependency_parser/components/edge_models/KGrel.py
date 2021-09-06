# This file is based on the BiaffineDependencyParser implemented in AllenNLP 0.8.4,
# which is licensed under the Apache License, Version 2.0.
# see https://github.com/allenai/allennlp/blob/030ef755aeeee7d05d119dcfe367f81bb26aed53/allennlp/models/biaffine_dependency_parser.py

#
# Copyright (c) 2021 Saarland University.
#
# This file is part of AM Parser
# (see https://github.com/coli-saar/am-parser/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
import math   # log
from allennlp.data import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.nn.util import get_device_of

from graph_dependency_parser.components.edge_models.base import EdgeModel
from graph_dependency_parser.components.edge_models.KG import KGEdges

import logging

logger = logging.getLogger(__name__)

# todo work in progress: needs testing, check for correctness and speed, refactor/comments


def get_positional_encodings(seqlen: int, dmodel: int) -> tuple:
    """
    Get positional encodings for all distances in a sentence (length seqlen)

    Uses the formula of Vaswani et al.  "Attention is all you need paper"
      PE_{pos,2i} = sin ( pos / 10000^{2i / dmodel} )
      PE_{pos,2i+1} = cos ( pos / 10000^{2i / dmodel} )
    where i is a concrete dimension in the position encoding,
    pos is the position (0..seqlen-1),
    dmodel is the dimensionality of the position encoding
    Implementation based on https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L65
    :param seqlen: length of the sequence: define possible relative distances
    :param dmodel: dimensionality of the distance encodings
    :return: ( encodings torch.Tensor with shape (2*seqlen-1, dmodel) ,
               distances torch.Tensor with shape (2*seqlen-1,)       )
    """
    assert seqlen > 0, f"Sequence must contain at least one token."
    assert dmodel % 2 == 0, f"Dimension of encoding (`dmodel`) must be a multiple of 2 (even number) but is {dmodel}"  # otherwise sine, cosine alternation
    # div_term representing: 1 / (10000^ (2i/dmodel))
    div_term = torch.exp(torch.arange(0, dmodel, step=2).float() * (
                -math.log(10000.0) / dmodel))  # using exp of the log math trick
    # div_term  shape: (dmodel/2)

    # possible distances  # todo turn into a function maybe?
    # positions -seqlen+1, ..., 0, ..., seqlen-1
    distances = torch.arange(start=-seqlen+1, end=seqlen, dtype=torch.float)  # e.g. tensor([-2., -1., 0., 1., 2.])
    assert distances.size() == (2*seqlen-1,)
    positions = distances.float().unsqueeze(1)
    # shape: (seqlen, 1)
    # todo do these encoding even/odd work wen full relative vec is entered?
    encodings = torch.zeros(2*seqlen-1, dmodel)  # shape: (seqlen, dmodel)
    encodings[:, 0::2] = torch.sin(positions * div_term)  # even dimensions of encoding: sinus
    encodings[:, 1::2] = torch.cos(positions * div_term)  # odd dimensions of encoding: cosinus
    # assert all tensors are requires-grad False
    # used in torch LM: encodings = encodings.unsqueeze(0).transpose(0,1)  # shape: (seqlen, 1, dmodel)
    return encodings, distances  # encodings shape: (2*seqlen-1, dmodel), distances shape: (2*seqlen-1,)


def get_distances_as_matrix(distance_vecs: torch.Tensor, seqlen: int) -> torch.Tensor:
    # distance_vecs: (2*seqlen-1, d_dim)
    # for seqlen==5 have vector for each of [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    assert(distance_vecs.size()[0] == distance_vecs.size()[0] == 2*seqlen-1)
    d_dim = distance_vecs.size()[1]
    m = torch.zeros(seqlen, seqlen, d_dim)
    for i in range(1, seqlen+1):
        # print(i)
        # print(m[i-1,:,:])
        m[i-1:,:,] = distance_vecs[seqlen-i:2*seqlen-i,:]
    # print(m)
    assert(m.size()==(seqlen, seqlen, d_dim))
    return m  # (seqlen, seqlen, d_dim)


@EdgeModel.register("kg_rel_edges")
class KGEdgesRel(KGEdges):  # inherits from EdgeModel
    """
    KG edge model but added relative position encoding

    For more details see the `kg_edges` model (`KG.py`) and the KG paper:
    based on a reimplementation of the edge model of the graph-based
    graph_dependency_parser by Kiperwasser and Goldberg (2016):
    https://aclweb.org/anthology/Q16-1023
    """
    def __init__(self, vocab: Vocabulary,
                 encoder_dim: int,
                 label_dim: int,
                 edge_dim: int,
                 edge_label_namespace: str,
                 dropout: float = 0.0,
                 activation: Activation = None,
                 dist_dim: int = 64,
                 ) -> None:
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
            dist_dim : ` int``, optional, (default = 64)
                Dimensionality of the relative distance encoding
        """
        super(KGEdgesRel, self).__init__(vocab=vocab,
                                         encoder_dim=encoder_dim,
                                         label_dim=label_dim,
                                         edge_dim=edge_dim,
                                         edge_label_namespace=edge_label_namespace,
                                         dropout=dropout,
                                         activation=activation)
        self.dist_dim = dist_dim
        # assert self.dist_dim > 0
        #edge existence:
        #these two matrices together form the feed forward network which takes the vectors of the two words in question and makes predictions from that
        #this is the trick described by Kiperwasser and Goldberg to make training faster.
        ##self.head_arc_feedforward = torch.nn.Linear(encoder_dim, edge_dim)
        ##self.child_arc_feedforward = torch.nn.Linear(encoder_dim, edge_dim, bias=False) #bias is already added by head_arc_feedforward
        self.dist_arc_feedforward = torch.nn.Linear(dist_dim, edge_dim, bias=False)

        ##self.arc_out_layer = torch.nn.Linear(edge_dim, 1, bias=False)  # K&G don't use a bias for the output layer
        return

    # todo overrides annotation?
    def edge_existence(self, encoded_text: torch.Tensor, mask : torch.LongTensor) -> torch.Tensor:
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
        # encoded_text: shape (batch_size, sequence length, encoding dim)
        # mask: shape (batch_size, sequence length)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self.head_arc_feedforward(encoded_text) #FF is from encoder_dim to edge_dim
        child_arc_representation = self.child_arc_feedforward(encoded_text)

        bs, sl, arc_dim = head_arc_representation.size()  # batch size, sequence length, arc_dim
        # print(f"bs= {bs}, sl={sl}, arc_dim={arc_dim}")

        # relative distance:
        # (1) get distances and encode each distance as vector (PE)
        encodings, dists = get_positional_encodings(seqlen=sl, dmodel=self.dist_dim)
        encodings = encodings.to(get_device_of(encoded_text))
        assert encodings.size() == (2*sl-1, self.dist_dim)
        assert dists.size() == (2*sl-1,)
        # (2) run FFN on these distance encodings todo batch size before or after?
        distance_hc_representation = self.dist_arc_feedforward(encodings)
        assert distance_hc_representation.size() == (2*sl-1, arc_dim)
        # (3) organize representations into a matrix of shape (bs, sl, sl, arc_dim)
        # unlike heads, childs don't repeat along rows/columns, but along the diagonals
        distances = get_distances_as_matrix(distance_vecs=distance_hc_representation,
                                            seqlen=sl)
        assert distances.size() == (sl, sl, arc_dim)
        distances = distances.to(get_device_of(encoded_text))  # todo can I just do encoded_text.device ?
        # shape: (sl, sl, arc_dim)
        distances = distances.repeat(bs, 1, 1, 1)
        # shape: (batch size, sl, sl, arc_dim)

        #now repeat the token representations to form a matrix:
        #shape (batch_size, sequence_length, sequence_length, arc_representation_dim)
        heads = head_arc_representation.repeat(1, sl, 1).reshape(bs, sl, sl, arc_dim) #heads in one direction
        deps = child_arc_representation.repeat(1, sl, 1).reshape(bs, sl, sl, arc_dim).transpose(1,2) #deps in the other direction

        # shape (batch_size, sequence_length, sequence_length, arc_representation_dim)
        combined = self.activation(heads + deps + distances) #now the feedforward layer that takes every pair of vectors (and their relative distance) for tokens is complete.
        #combined now represents the activations in the hidden layer of the MLP.
        # now go from hidden (arc_dim) to output layer (1)
        edge_scores = self.arc_out_layer(combined).squeeze(3) #last dimension is now 1, remove it
        # (batch size, seq length, seq length)

        #mask out stuff for padded tokens:
        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        edge_scores = edge_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        return edge_scores # (batch size, seq length, seq length)

