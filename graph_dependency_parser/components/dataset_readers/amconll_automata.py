#
# Copyright (c) 2020 Saarland University.
#
# This file is part of AM Parser
# (see https://github.com/coli-saar/am-parser/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import io



from graph_dependency_parser.components.dataset_readers.rule_index_field import RuleIndexField

from jnius import autoclass

import zipfile
from typing import Dict, Tuple, List, Any
import logging

from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from graph_dependency_parser.components.dataset_readers.rule_mask_field import RuleMaskField
from graph_dependency_parser.inside_maximization.scyjava import to_python

from graph_dependency_parser.components.dataset_readers.supertag_map_field import SupertagMapField
from graph_dependency_parser.components.dataset_readers.amconll_tools import parse_amconll, AMSentence


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("amconll_automata")
class AMConllAutomataDatasetReader(DatasetReader):
    """
    Reads a file in amconll format containing AM dependency trees.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False, fraction: float = 1.0, only_read_fraction_if_train_in_filename : bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.fraction = fraction
        self.only_read_fraction_if_train_in_filename = only_read_fraction_if_train_in_filename

    def _read_one_file(self, formalism:str, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        AutomataZipReader = autoclass('de.saar.coli.amtools.decomposition.AutomataZipReader')
        automata_zip_reader = AutomataZipReader(file_path)
        if self.fraction < 0.9999 and (not self.only_read_fraction_if_train_in_filename or (self.only_read_fraction_if_train_in_filename and "train" in file_path)):
            with zipfile.ZipFile(file_path) as z:
                with io.TextIOWrapper(z.open("corpus.amconll")) as amconll_file:
                    logger.info("Reading a fraction of "+str(self.fraction)+" of the AM dependency trees from amconll dataset at: %s", file_path)
                    sents = list(parse_amconll(amconll_file))
                    for i, am_sentence in  enumerate(sents):
                        if i <= len(sents) * self.fraction:
                            yield self._read_one_sentence(formalism, automata_zip_reader, i, am_sentence)
        else:
            with zipfile.ZipFile(file_path) as z:
                with io.TextIOWrapper(z.open("corpus.amconll")) as amconll_file:
                    logger.info("Reading AM dependency trees from amconll dataset at: %s", file_path)
                    for i, am_sentence in enumerate(parse_amconll(amconll_file)):
                        yield self._read_one_sentence(formalism, automata_zip_reader, i, am_sentence)

    def _read_one_sentence(self, formalism:str, automata_zip_reader, position_in_corpus:int, am_sentence: AMSentence):
        automaton = automata_zip_reader.readAutomaton(position_in_corpus)
        supertag_map = automata_zip_reader.readSupertagMap(position_in_corpus, automaton)
        edge_map = automata_zip_reader.readEdgeMap(position_in_corpus, automaton)
        return self.text_to_instance(formalism, position_in_corpus, am_sentence, supertag_map, edge_map, automaton)

    @overrides
    def _read(self, file_paths: List[List[str]]):
        for per_formalism in file_paths:
            assert len(per_formalism)==2, f"list per formalism must have length two and must be structured as [task_name, path_to_data], got {per_formalism}"
            formalism, path = per_formalism
            for instance in self._read_one_file(formalism, path):
                yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         formalism: str,
                         position_in_corpus : int,
                         am_sentence: AMSentence,
                         supertag_map,
                         edge_map,
                         automaton) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        formalism : str.
            The formalism of this instance (e.g. DM, PSD, ...)
        position_in_corpus : ``int``, required.
            The index of this sentence in the corpus.
        am_sentence : ``AMSentence``, required.
            The words in the sentence to be encoded.

        Returns
        -------
        An instance containing words, pos tags, dependency edge labels, head
        indices, supertags and lexical labels as fields.
        """
        fields: Dict[str, Field] = {}

        all_rules_in_bottom_up_order = automaton.getAllRulesInBottomUpOrder()
        # for rule in to_python(rule_iterator):
        #     print(rule.toString(automaton))
        #     if supertag_map.keySet().contains(rule):
        #         print(f"Word position: {supertag_map.get(rule).left}")
        #     else:
        #         print(f"Edge position: {edge_map.get(rule).left.right}")
        tokens = TextField([Token(w) for w in am_sentence.get_tokens(shadow_art_root=True)], self._token_indexers)
        fields["words"] = tokens
        fields["pos_tags"] = SequenceLabelField(am_sentence.get_pos(), tokens, label_namespace="pos")
        fields["ner_tags"] = SequenceLabelField(am_sentence.get_ner(), tokens, label_namespace="ner_labels")
        fields["lemmas"] = SequenceLabelField(am_sentence.get_lemmas(), tokens, label_namespace="lemmas")
        fields["lexlabels"] = SequenceLabelField(am_sentence.get_lexlabels(), tokens, label_namespace=formalism+"_lex_labels")
        fields["head_indices"] = SequenceLabelField(am_sentence.get_heads(),tokens,label_namespace="head_index_tags")
        fields["rule_index"] = RuleIndexField(supertag_map, edge_map, all_rules_in_bottom_up_order, len(tokens) + 1, # +1 for artificial root
                                              supertag_namespace=formalism+"_supertag_labels",
                                              edge_namespace=formalism+"_head_tags")
        fields["rule_mask"] = RuleMaskField(all_rules_in_bottom_up_order, supertag_map, len(tokens) + 1)
        fields["metadata"] = MetadataField({"words": am_sentence.words, "attributes": am_sentence.attributes,
                                            "formalism": formalism, "position_in_corpus" : position_in_corpus,
                                            "token_ranges": am_sentence.get_ranges(),
                                            "is_annotated": am_sentence.is_annotated(),
                                            "max_state_id_plus_one": automaton.getStateInterner().getNextIndex(),
                                            "final_states": automaton.getFinalStates(),
                                            "all_rules_in_bottom_up_order": all_rules_in_bottom_up_order})
        # checking rule identity across maps and automaton
        # print("Rules in supertag_map:")
        # for rule in to_python(supertag_map.keySet()):
        #     print(rule)
        #     print(rule.toString(automaton))
        # print("Rules in edge_map:")
        # for rule in to_python(edge_map.keySet()):
        #     print(rule)
        #     print(rule.toString(automaton))
        # print("Rules in automaton:")
        # for rule in to_python(automaton.getRuleSet()):
        #     print(rule)
        #     print(rule.toString(automaton))
        # print("Modifying rules in supertag_map:")
        # for rule in to_python(supertag_map.keySet()):
        #     rule.setWeight(0.5)
        #     print(rule)
        #     print(rule.toString(automaton))
        # print("Modifying rules in edge_map:")
        # for rule in to_python(edge_map.keySet()):
        #     rule.setWeight(0.2)
        #     print(rule)
        #     print(rule.toString(automaton))
        # print("Modified rules in automaton:")
        # for rule in to_python(automaton.getRuleSet()):
        #     print(rule)
        #     print(rule.toString(automaton))
        return Instance(fields)

    @staticmethod
    def restore_order(instances : List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tries to restore the order that was used when the instances were read.
        :param instances:
        :return:
        """
        return sorted(instances, key=lambda d: d["position_in_corpus"])
