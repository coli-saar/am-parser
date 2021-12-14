from typing import Optional

import torch


class BatchedListofList:
    """
    A list of lists for each batch element. Basically a tensor of shape (batch_size, outer_size, inner_size)
    """
    def __init__(self, batch_size : int, outer_size: int, inner_size : int, device = None, stacktype : Optional[torch.dtype] = torch.long):
        self.lol = torch.zeros(batch_size, outer_size, inner_size, dtype=stacktype, device=device)
        self.ptr = torch.zeros(batch_size, outer_size, dtype=torch.long, device=device)
        self.batch_range = torch.arange(batch_size, dtype=torch.long, device=device)

    def outer_index(self, indices):
        return self.lol[self.batch_range, indices]

    def append(self, outer_indices, vector, mask):
        """

        :param outer_indices: shape (outer_size,)
        :param vector: shape (batch_size,)
        :param mask:
        :return:
        """
        mask = mask.long()
        self.lol[self.batch_range, outer_indices, self.ptr[self.batch_range, outer_indices]] = (1-mask)*self.lol[self.batch_range, outer_indices, self.ptr[self.batch_range, outer_indices]] \
                                                                                               + mask * vector
        self.ptr[self.batch_range, outer_indices] += mask


