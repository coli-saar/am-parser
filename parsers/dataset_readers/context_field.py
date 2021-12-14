from typing import Dict, List

import torch
from allennlp.data import Field, DataArray, Vocabulary
from allennlp.data.fields import ListField

SEPARATOR = "@--@"

class ContextField(Field):
    """
    A field with multiple named subfields that themselves are ListFields.
    """

    def __init__(self, data : Dict[str, ListField]) -> None:
        self.data = data

    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        sub_padding_lengths = dict()
        for k,v in padding_lengths.items():
            subfname = k.split(SEPARATOR)[0]
            rest = '_'.join(k.split(SEPARATOR)[1:])
            if subfname not in sub_padding_lengths:
                sub_padding_lengths[subfname] = dict()
            sub_padding_lengths[subfname][rest] = v

        return {name: val.as_tensor(sub_padding_lengths[name]) for name, val in self.data.items()}

    def empty_field(self) -> 'Field':
        return ContextField(dict())

    def get_padding_lengths(self) -> Dict[str, int]:
        ret = {}
        for name, subf in self.data.items():
            for k, v in subf.get_padding_lengths().items():
                ret[name+SEPARATOR+k] = v
        return ret

    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:
        ret = dict()
        for name, subfield in self.data.items():
            ret[name] = subfield.batch_tensors([ elem[name] for elem in tensor_list])
        return ret

    def index(self, vocab: Vocabulary):
        for subf in self.data.values():
            subf.index(vocab)