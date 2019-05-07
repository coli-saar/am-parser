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
        self.hinge_loss = torch.nn.MultiLabelMarginLoss(reduction="none")

    def loss(self, edge_label_logits:torch.Tensor, mask:torch.Tensor, head_tags:torch.Tensor) -> torch.Tensor:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        edge_label_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            that contains raw predictions for incoming edge labels
        head_tags : ``torch.Tensor``, required.
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
        head_tags = head_tags[:,1:]
        mask = mask[:,1:]

        batch_size, sequence_length, num_head_tags = edge_label_logits.size()
        head_tags_r = head_tags.reshape(batch_size * sequence_length).unsqueeze(1) #shape (batch_size * seq len, 1)

        #We want to use a hinge loss for multiple classes but pytorch only offers one where an instance can have multiple labels
        #See here: https://pytorch.org/docs/stable/nn.html#torch.nn.MultiLabelMarginLoss
        #To denote classes that do not apply to an instance, we use -1 as the target

        padding = -torch.ones((batch_size*sequence_length,num_head_tags-1),device=get_device_of(head_tags),dtype=torch.long)
        targets = torch.cat([head_tags_r, padding],dim=1) #shape (batch_size*seq len, num_head_tag_logits), the last num_head_tag_logits-1 dimensions on axis 2 are filled with -1

        head_tag_logits_r = edge_label_logits.reshape(batch_size * sequence_length, num_head_tags)

        mask_r = mask.reshape((batch_size*sequence_length)).float()

        loss = (self.hinge_loss(head_tag_logits_r,targets) * mask_r).sum()

        if self.normalize_wrt_seq_len:
            valid_positions = mask_r.sum()
            loss /= valid_positions
        return loss