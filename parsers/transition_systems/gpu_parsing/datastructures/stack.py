from typing import Optional

import torch


class BatchedStack:
    def __init__(self, batch_size : int, max_capacity: int, device = None, stacktype : Optional[torch.dtype] = torch.long):
        self.max_capacity = max_capacity
        self.stack = torch.zeros(batch_size, max_capacity, dtype=stacktype, device=device)
        self.stack_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.batch_range = torch.arange(batch_size, dtype=torch.long, device=device)
        self.done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def depth(self) -> torch.Tensor:
        return self.stack_ptr.clone()

    def push(self, vector : torch.Tensor, mask : torch.Tensor) -> None:
        """
        Pushes a vector to the stack, but only for those cases where the mask is 1.
        :param vector: shape (batch_size,)
        :param mask: shape (batch_size,)
        :return:
        """
        mask = mask.long()
        self.stack_ptr += mask
        self.stack[self.batch_range, self.stack_ptr] = (1-mask)*self.stack[self.batch_range, self.stack_ptr] + mask * vector

        if torch.any(self.stack_ptr >= self.max_capacity):
            raise ValueError("Stack overflow")

    def peek(self):
        return self.stack[self.batch_range, self.stack_ptr]

    def get_done(self):
        """
        Return places where stack has become empty through pop operations.
        :return:
        """
        return self.done_mask

    def pop_wo_peek(self, mask : torch.Tensor) -> None:
        """
        Remove the top of the stack for those positions where mask[i] = 1
        :param mask: shape (batch_size,)
        :return:
        """
        self.stack_ptr -= mask.long()
        self.done_mask |= (self.stack_ptr == 0)
        if torch.any(self.stack_ptr < 0):
            raise ValueError("Stack empty")

    def is_empty(self) -> torch.Tensor:
        return self.stack_ptr == 0

    def pop(self, mask) -> torch.Tensor:
        """
        Combination of peek and pop_wo_peek()
        :param mask: where to pop things from the stack.
        :return:
        """
        r = self.peek()
        self.pop_wo_peek(mask)
        return r

    def pop_and_push_multiple(self, elements, pop_mask, push_mask, reverse: Optional[bool] = False) -> None:
        """
        Pops Pushes elements on stack
        :param reverse:
        :param elements: Tensor of shape (batch_size, some_len)
        :param mask: Tensor of shape (batch_size, som_len)
        """
        self.stack_ptr -= pop_mask.long()
        if torch.any(self.stack_ptr < 0):
            raise ValueError("Stack empty")

        mask = push_mask #.long()
        neg_mask = ~mask
        if reverse:
            r = range(elements.shape[1])
        else:
            r = range(elements.shape[1]-1,-1,-1)

        something_in_that_dim = torch.any(mask, dim=0).cpu().numpy() #shape (some_len,)
        for i in r:
            if something_in_that_dim[i]:
                m = mask[:, i]
                vector = elements[:, i]
                self.stack_ptr += m
                self.stack[self.batch_range, self.stack_ptr] = neg_mask[:,i] * self.stack[self.batch_range, self.stack_ptr] +  m * vector

        self.done_mask |= (self.stack_ptr == 0)
