from typing import Dict, List

import torch

from allennlp.data.fields.metadata_field import DataArray, Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import pad_sequence_to_length

from overrides import overrides

from graph_dependency_parser.components.dataset_readers.amconll_tools import AMSentence
from graph_dependency_parser.inside_maximization.scyjava import to_python

class RuleIndexField(Field):

    def __init__(self, supertag_map, edge_map, rule_iterator, sentence_length, supertag_namespace: str, edge_namespace):
        """

        :param supertag_map: a java HashMap<Rule, Pair<Integer, String>> that maps rules to word positions and supertags
        """

        self.rule_iterator = []
        self.supertag_map = dict()
        self.edge_map = dict()
        for rule in to_python(rule_iterator):
            self.rule_iterator.append(rule)
            if supertag_map.keySet().contains(rule):
                pair = supertag_map.get(rule)
                self.supertag_map[rule] = (pair.left, pair.right)
            elif edge_map.keySet().contains(rule):
                pair = edge_map.get(rule)
                self.edge_map[rule] = (pair.left.left, pair.left.right, pair.right)
            else:
                raise Exception(f"Rule not found in either supertag or edge map: {rule}")

        if not edge_map.keySet().size() + supertag_map.keySet().size() == len(self.rule_iterator):
            raise Exception(f"Sizes of edge map + supertag map ({edge_map.keySet().size() + supertag_map.keySet().size()})"
                            f"don't match automaton rule count ({len(self.rule_iterator)})")
        self.supertag_namespace = supertag_namespace
        self.edge_namespace = edge_namespace
        self.sentence_length = sentence_length
        self.index_list = None
        self.supertag_vocab_size = None

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for pair in self.supertag_map.values():
            counter[self.supertag_namespace][pair[1]] += 1
        for triple in self.edge_map.values():
            counter[self.edge_namespace][triple[2]] += 1
        counter[self.supertag_namespace][AMSentence.get_bottom_supertag()] += 1
        counter[self.edge_namespace]["ROOT"] += 1
        counter[self.edge_namespace]["IGNORE"] += 1

    @overrides
    def index(self, vocab: Vocabulary):
        # creates an index_list which contains, for each rule in self.rule_iterator, the index corresponding to the rule
        # in the following vector: namely stacking all possible supertags for each word on top of each other and
        # stacking all possible (incoming) edge labels for each word on top of that. So that vector has
        # size sentence_length * supertag_vocab_size + sentence_length * edge_vocab_size.
        # We also store supertag_vocab_size, for future reference during padding

        # diagnostic printing for something else
        # print(vocab.get_index_to_token_vocabulary(namespace=self.supertag_namespace))
        # print(vocab.get_index_to_token_vocabulary(namespace=self.edge_namespace))

        self.index_list = []
        self.supertag_vocab_size = vocab.get_vocab_size(namespace=self.supertag_namespace)
        edge_vocab_size = vocab.get_vocab_size(namespace=self.edge_namespace)
        covered_word_positions = set()
        word_positions_with_incoming_edge = set()
        for rule in self.rule_iterator:
            if rule in self.supertag_map:
                pair = self.supertag_map[rule]
                word_position = pair[0]#+1 # +1 to account for artificial root
                covered_word_positions.add(word_position)
                supertag_id = vocab.get_token_index(pair[1], namespace=self.supertag_namespace)
                index = word_position*self.supertag_vocab_size+supertag_id
            elif rule in self.edge_map:
                triple = self.edge_map[rule]
                child_position = triple[1]# + 1 # +1 to account for artificial root
                word_positions_with_incoming_edge.add(child_position)
                edge_label_id = vocab.get_token_index(triple[2], namespace=self.edge_namespace)
                index = self.sentence_length * self.supertag_vocab_size + child_position * edge_vocab_size + edge_label_id
            else:
                print(self.edge_map)
                print(self.supertag_map)
                print(rule)
                raise Exception(f"Rule not found in either supertag or edge map: {rule}")
            self.index_list.append(index)
        print(f"root id: {covered_word_positions.difference(word_positions_with_incoming_edge)}")
        bottom_id = vocab.get_token_index(AMSentence.get_bottom_supertag(), namespace=self.supertag_namespace)
        ignore_id = vocab.get_token_index("IGNORE", namespace=self.edge_namespace)
        root_id = vocab.get_token_index("ROOT", namespace=self.edge_namespace)
        for pos in range(1, self.sentence_length):  # start at 1 to skip artificial root
            if pos not in covered_word_positions:
                self.index_list.append(pos*self.supertag_vocab_size + bottom_id)
                self.index_list.append(self.sentence_length * self.supertag_vocab_size + pos * edge_vocab_size + ignore_id)
        for pos in covered_word_positions.difference(word_positions_with_incoming_edge):
            self.index_list.append(self.sentence_length * self.supertag_vocab_size + pos * edge_vocab_size + root_id)
        print(self.index_list)



    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        ret = dict()
        ret["sentence_length"] = self.sentence_length
        ret["rule_count"] = len(self.index_list)
        return ret

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        sentence_length_diff = padding_lengths["sentence_length"] - self.sentence_length
        padded_rule_count = padding_lengths["rule_count"]
        new_index_list = []
        if sentence_length_diff > 0:
            for index in enumerate(self.index_list):
                if index >= self.sentence_length*self.supertag_vocab_size:
                    new_index_list.append(index+sentence_length_diff*self.supertag_vocab_size)
                else:
                    new_index_list.append(index)
        padded_indices = pad_sequence_to_length(self.index_list, padded_rule_count) # TODO will need to mask somewhere
        return torch.LongTensor(padded_indices)

    def __str__(self) -> str:
        return f"RuleIndexField with supertag map {self.supertag_map},\n" \
               f"edge map {self.edge_map}\n" \
               f"index list {self.index_list}"
