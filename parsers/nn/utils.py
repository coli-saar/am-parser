from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from allennlp.nn.util import get_device_of

import numpy as np

def get_device_id(t : torch.Tensor):
    d = get_device_of(t)
    if d < 0:
        return None
    return d

def index_tensor_dict(d: Dict[str, torch.Tensor], i : int) -> Dict[str, torch.Tensor]:
    """
    Index tensor dict by first dimension.
    :param d:
    :return:
    """
    return {k : v[i] for k,v in d.items()}

def batch_and_pad_tensor_dict(tensor_dicts : List[Dict[str, torch.Tensor]]) -> Dict[str,torch.Tensor]:
    """
    Batch and pad a tensor dict with 0s.
    :param tensor_dicts:
    :return:
    """
    key_to_tensors: Dict[str, List[torch.Tensor]] = dict()
    key_to_dim : Dict[str, List[np.array]] = dict()
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            if key not in key_to_tensors:
                key_to_tensors[key] = []
                key_to_dim[key] = []

            key_to_tensors[key].append(tensor)
            key_to_dim[key].append(np.array(tensor.shape))

    key_to_max_dim = dict()
    for key in key_to_dim:
        key_to_max_dim[key] = np.max(np.stack(key_to_dim[key]), axis=0)

    ret = dict()
    for key in key_to_tensors:
        liste = []
        for tensor in key_to_tensors[key]:
            diff_dim = key_to_max_dim[key] - np.array(tensor.shape) # where to add additional elements
            add_elements = [ 0 if i % 2 == 0 else diff_dim[i//2] for i in range(2*diff_dim.shape[0],0,-1)]
            liste.append(F.pad(tensor, add_elements, "constant", 0))

        ret[key] = torch.stack(liste)
    return ret

def expand_tensor_dict(td : Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    batch_size = next(iter(td.values())).shape[0]
    return [index_tensor_dict(td, i) for i in range(batch_size)]

def move_tensor_dict(td : Dict[str, torch.Tensor], device : Optional[int]) -> Dict[str, torch.Tensor]:
    return {k: t.to(device) for k,t in td.items()}