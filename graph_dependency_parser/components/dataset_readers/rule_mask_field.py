from typing import Dict, List

import torch

from allennlp.data.fields.metadata_field import DataArray, Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import pad_sequence_to_length

from overrides import overrides

from graph_dependency_parser.components.dataset_readers.amconll_tools import AMSentence
from graph_dependency_parser.inside_maximization.scyjava import to_python


class RuleMaskField(Field):

    def __init__(self, rule_iterator, supertag_map, sentence_length):
        """
        :param supertag_map: a java HashMap<Rule, Pair<Integer, String>> that maps rules to word positions and supertags
        """
        # we just need to count how many rules there are (including 'fake rules' for not covered words and ROOT)
        self.rule_count = 1 # always exactly one 'fake rule' for ROOT edge
        covered_word_positions = set()
        for rule in to_python(rule_iterator):
            self.rule_count += 1
            if supertag_map.keySet().contains(rule):
                word_position = supertag_map.get(rule).left
                covered_word_positions.add(word_position)
        # now count how many 'fake rules' for not covered words there are:
        for pos in range(1, sentence_length):
            if pos not in covered_word_positions:
                self.rule_count += 2  # one 'fake rule' for bottom supertag, one for IGNORE edge label

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        ret = dict()
        ret["rule_count"] = self.rule_count
        return ret

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        padded_rule_count = padding_lengths["rule_count"]
        mask = pad_sequence_to_length([1 for _ in range(self.rule_count)], padded_rule_count)
        return torch.BoolTensor(mask)

