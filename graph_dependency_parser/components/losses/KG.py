#
# Copyright (c) 2020 Saarland University.
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
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask

from typing import Dict, Optional, Tuple, Any, List
import numpy as np

import torch

from graph_dependency_parser.components.cle import cle_loss
from graph_dependency_parser.components.losses.base import EdgeExistenceLoss, EdgeLabelLoss


@EdgeExistenceLoss.register("kg_edge_loss")
class KGExistenceLoss (EdgeExistenceLoss):

    """
    Kiperwasser & Goldberg Hinge Loss.
    """


    def loss(self, edge_scores: torch.Tensor,
                            head_indices: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the edge loss for a sequence given gold head indices.

        Parameters
        ----------
        edge_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        """
        lengths = mask.data.sum(dim=1).long()

        return cle_loss(edge_scores,lengths,head_indices,self.normalize_wrt_seq_len)





@EdgeLabelLoss.register("kg_label_loss")
class KGLabelLoss(EdgeLabelLoss):
    """
    A hinge loss for edge labels as used by Kiperwasser and Goldberg
    """

    def __init__(self, normalize_wrt_seq_len : bool = True):
        super().__init__()
        self.normalize_wrt_seq_len = normalize_wrt_seq_len
        self.hinge_loss = torch.nn.MultiMarginLoss(reduction='none')

    def loss(self, edge_label_logits:torch.Tensor, mask:torch.Tensor, gold_edge_labels:torch.Tensor) -> torch.Tensor:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        edge_label_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, num_edge_labels),
            that contains raw predictions for incoming edge labels
        gold_edge_labels : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        loss : ``torch.Tensor``, required.
            The hinge edge label loss.
        """
        #Remove artificial root:
        edge_label_logits = edge_label_logits[:, 1:, :]
        gold_edge_labels = gold_edge_labels[:, 1:]
        mask = mask[:,1:]

        batch_size, sequence_length, num_edge_labels = edge_label_logits.size()
        #We have to reshape the label predictions to be of shape (some batch size, num of edge labels)
        #and the gold edge labels to be of shape (some batch size)
        #see https://pytorch.org/docs/stable/nn.html#torch.nn.MultiMarginLoss for details

        gold_edge_labels_r = gold_edge_labels.reshape(batch_size * sequence_length)

        edge_label_logits_r = edge_label_logits.reshape(batch_size * sequence_length, num_edge_labels)

        mask_r = mask.reshape((batch_size*sequence_length)).float()

        loss = (self.hinge_loss(edge_label_logits_r,gold_edge_labels_r) * mask_r).sum()

        if self.normalize_wrt_seq_len:
            valid_positions = mask_r.sum()
            loss /= valid_positions
        return loss