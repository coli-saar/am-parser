from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask

from typing import Dict, Optional, Tuple, Any, List

import torch

from graph_dependency_parser.components.losses.base import EdgeLoss, EdgeExistenceLoss, EdgeLabelLoss

@EdgeExistenceLoss.register("dm_edge_loss")
class DMLoss (EdgeExistenceLoss):
    """
    Dozat & Manning - Loss.
    """

    def loss(self, edge_scores: torch.Tensor,
                            head_indices: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the edge loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        edge_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = edge_scores.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(edge_scores)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = masked_log_softmax(edge_scores,
                                                   mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length, get_device_of(edge_scores))
        child_index = timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum()
        if self.normalize_wrt_seq_len:
            arc_nll /= valid_positions.float()
        return arc_nll

@EdgeLabelLoss.register("dm_label_loss")
class DMLabelLoss(EdgeLabelLoss):

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
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the edge label loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = edge_label_logits.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(edge_label_logits)).unsqueeze(1)

        # shape (batch_size, sequence_length, num_head_tags)
        normalised_edge_label_logits = masked_log_softmax(edge_label_logits,
                                                        mask.unsqueeze(-1)) * float_mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length, get_device_of(edge_label_logits))
        child_index = timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        # shape (batch_size, sequence_length)
        tag_loss = normalised_edge_label_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        if self.normalize_wrt_seq_len:
            return -tag_loss.sum()  / valid_positions.float()
        else:
            return -tag_loss.sum()