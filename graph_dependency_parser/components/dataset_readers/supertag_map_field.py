from typing import Dict, List

from allennlp.data.fields.metadata_field import DataArray, Field
from allennlp.data.vocabulary import Vocabulary

from overrides import overrides

from graph_dependency_parser.inside_maximization.scyjava import to_python

class SupertagMapField(Field):

    def __init__(self, supertag_map, label_namespace: str = 'labels'):
        """

        :param supertag_map: a java HashMap<Rule, Pair<Integer, String>> that maps rules to word positions and supertags
        """
        self.map = dict()
        for rule in to_python(supertag_map.keySet()):
            pair = supertag_map.get(rule)
            self.map[rule] = (pair.left, pair.right)
        self._label_namespace = label_namespace
        self.indexed_map = None


    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for pair in self.map.values():
            counter[self._label_namespace][pair[1]] += 1

    @overrides
    def index(self, vocab: Vocabulary):
        self.indexed_map = dict()
        for key in self.map.keys():
            pair = self.map[key]
            self.indexed_map[key] = (pair[0], vocab.get_token_index(pair[1], self._label_namespace))

    # like in MetadataField
    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    # analog to MetadataField
    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        return self.indexed_map

    # like in MetadataField
    @classmethod
    @overrides
    def batch_tensors(cls, tensor_list: List[DataArray]) -> List[DataArray]:  # type: ignore
        return tensor_list

    # analog to MetadataField
    def __str__(self) -> str:
        return f"SupertagMapField (print field.map to see specific information)."
