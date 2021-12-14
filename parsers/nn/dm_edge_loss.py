import torch
from allennlp.nn.util import get_range_vector, get_device_of, masked_log_softmax


def loss(edge_scores: torch.Tensor,
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

    arc_nll = -arc_loss.sum()
    return arc_nll