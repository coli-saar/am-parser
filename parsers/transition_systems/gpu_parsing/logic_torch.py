from typing import Tuple, Set

import torch


def make_bool_multipliable(t : torch.Tensor) -> torch.Tensor:
    if t.is_cuda:
        if not (t.dtype == torch.float or t.dtype == torch.float16):
            return t.half() #float 16 is good enough, I hope.
    else:
        if not (t.dtype == torch.int32 or t.dtype == torch.long or t.dtype == torch.int16):
            return t.int()
    return t


def are_eq(a : torch.Tensor, b : torch.Tensor) -> torch.BoolTensor:
    if a.dtype == torch.float or a.dtype == torch.float16 or b.dtype == torch.float or b.dtype == torch.float16:
        return torch.abs(a-b) < 0.00001
    else:
        return a == b


def index_or(batched_set : torch.BoolTensor, mapping : torch.BoolTensor) -> torch.BoolTensor:
    """
    batched_set : shape (batch_size, set capacity)
    mapping : shape (set capacity, "constants")
    returns a bool tensor R of shape (batch_size, "constants")
    R[b,c] = True iff \exists l, batched_set[b,l] AND mapping[l,c]
    """
    result = batched_set @ mapping #shape (batch_size, "constants")
    return result > 0


def batched_index_or(batched_set : torch.BoolTensor, mapping : torch.BoolTensor) -> torch.BoolTensor:
    """
    batched_set : shape (batch_size, set capacity)
    mapping : shape (batch_size, set capacity, "constants")
    returns a bool tensor R of shape (batch_size, "constants")
    R[b,c] = True iff \exists l, batched_set[b,l] AND mapping[b,l,c]
    """
    result = (torch.bmm(batched_set.unsqueeze(1), mapping)).squeeze(1) #shape (batch_size, "lexical types")
    return result > 0


def debug_to_set(t : torch.BoolTensor) -> Set[int]:
    return [{ i for i,x in enumerate(batch) if x} for batch in t.numpy()]