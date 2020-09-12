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
        print("rule info")
        for rule in to_python(rule_iterator):
            self.rule_iterator.append(rule)
            if supertag_map.keySet().contains(rule):
                pair = supertag_map.get(rule)
                self.supertag_map[rule] = (pair.left, pair.right)
                print(f"st {pair.left}")
            elif edge_map.keySet().contains(rule):
                pair = edge_map.get(rule)
                self.edge_map[rule] = (pair.left.left, pair.left.right, pair.right)
                print(f"el {pair.left.right}")
            else:
                raise Exception(f"Rule not found in either supertag or edge map: {rule}")

        if not edge_map.keySet().size() + supertag_map.keySet().size() == len(self.rule_iterator):
            raise Exception(f"Sizes of edge map + supertag map ({edge_map.keySet().size() + supertag_map.keySet().size()})"
                            f"don't match automaton rule count ({len(self.rule_iterator)})")
        self.supertag_namespace = supertag_namespace
        self.edge_namespace = edge_namespace
        self.sentence_length = sentence_length
        self.index_list = None
        self.vocab = None  # will need to store vocab later for correct shifting of indices during padding
        self.supertag_index_max = None

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
        self.vocab = vocab  # store vocab for correct shifting of indices during padding later
        supertag_vocab_size = vocab.get_vocab_size(namespace=self.supertag_namespace)
        self.supertag_index_max = self.sentence_length * supertag_vocab_size  # note that supertag_vocab_size is
        # not globally correct, but we can use it locally (updated to global correct value in as_tensor)
        edge_vocab_size = vocab.get_vocab_size(namespace=self.edge_namespace)
        print(f"edge vocab: {vocab.get_index_to_token_vocabulary(namespace=self.edge_namespace)}")
        print(f"st vocab: {vocab.get_index_to_token_vocabulary(namespace=self.supertag_namespace)}")
        covered_word_positions = set()
        word_positions_with_incoming_edge = set()
        for rule in self.rule_iterator:
            if rule in self.supertag_map:
                pair = self.supertag_map[rule]
                word_position = pair[0]#+1 # +1 to account for artificial root
                covered_word_positions.add(word_position)
                supertag_id = vocab.get_token_index(pair[1], namespace=self.supertag_namespace)
                index = word_position*supertag_vocab_size+supertag_id
            elif rule in self.edge_map:
                triple = self.edge_map[rule]
                child_position = triple[1]# + 1 # +1 to account for artificial root
                word_positions_with_incoming_edge.add(child_position)
                edge_label_id = vocab.get_token_index(triple[2], namespace=self.edge_namespace)
                index = self.sentence_length * supertag_vocab_size + child_position * edge_vocab_size + edge_label_id
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
                self.index_list.append(pos*supertag_vocab_size + bottom_id)
                self.index_list.append(self.sentence_length * supertag_vocab_size + pos * edge_vocab_size + ignore_id)
        for pos in covered_word_positions.difference(word_positions_with_incoming_edge):
            self.index_list.append(self.sentence_length * supertag_vocab_size + pos * edge_vocab_size + root_id)
        # print("index list")
        # for index in self.index_list:
        #     print(index)



    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        ret = dict()
        ret["sentence_length"] = self.sentence_length
        ret["rule_count"] = len(self.index_list)
        return ret

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        padded_sentence_length = padding_lengths["sentence_length"]
        padded_rule_count = padding_lengths["rule_count"]
        supertag_vocab_size = self.vocab.get_vocab_size(namespace=self.supertag_namespace)  # now globally correct
        new_index_list = []
        print(f"self.supertag_index_max: {self.supertag_index_max}")
        print(f"padded_sentence_length: {padded_sentence_length}")
        print(f"supertag_vocab_size: {supertag_vocab_size}")
        print("indices before:")
        for index in self.index_list:
            print(index)
            # print(f"index: {index}")
            # print(f"sent length: {self.sentence_length}")
            # print(f"st vocab size: {supertag_vocab_size}")
            if index >= self.supertag_index_max:
                new_index_list.append(index + (padded_sentence_length - self.sentence_length) * supertag_vocab_size)
            else:
                new_index_list.append(index)
        padded_indices = pad_sequence_to_length(new_index_list, padded_rule_count)
        print("indices after:")
        for index in padded_indices:
            print(index)
        return torch.LongTensor(padded_indices)

    def __str__(self) -> str:
        return f"RuleIndexField with supertag map {self.supertag_map},\n" \
               f"edge map {self.edge_map}\n" \
               f"index list {self.index_list}"
