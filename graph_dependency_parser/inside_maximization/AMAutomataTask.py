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
from time import time
from typing import Dict, Optional, List, Any

import numpy
import torch

from allennlp.common import Registrable
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import InputVariationalDropout
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, get_device_of
from allennlp.training.metrics import CategoricalAccuracy, AttachmentScores
from allennlp.training.metrics.average import Average
from overrides import overrides
from torch.nn import Module

from jnius import autoclass

from graph_dependency_parser.components.cle import cle_decode, find_root
from graph_dependency_parser.components.copier import Copier
from graph_dependency_parser.components.dataset_readers.amconll_tools import AMSentence
from graph_dependency_parser.components.edge_models import EdgeModel
from graph_dependency_parser.components.evaluation.predictors import AMconllPredictor, Evaluator
from graph_dependency_parser.components.losses import EdgeLoss
from graph_dependency_parser.components.losses.supertagging import SupertaggingLoss
from graph_dependency_parser.components.supertagger import Supertagger

import torch.nn.functional as F

import logging

from graph_dependency_parser.inside_maximization.scyjava import to_python

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

POS_TO_IGNORE = {'``', "''", ':', ',', '.', 'PU', 'PUNCT', 'SYM'}

PyjniusHelper = autoclass('de.saar.coli.amtools.decomposition.PyjniusHelper')


def none_or_else(exp, val):
    if exp is not None:
        return val
    return None


def mix_loss(coefficient, tensor):
    if abs(coefficient - 1.0) < 0.0001:
        return tensor
    return coefficient * tensor


class AMAutomataTask(Model):
    """
    A class that implements a task-specific model. It conceptually belongs to a formalism or corpus.
    """
    loss_names = ["edge_existence", "edge_label", "supertagging", "lexlabel"]

    def __init__(self, vocab: Vocabulary,
                 name: str,
                 edge_model: EdgeModel,
                 loss_function: EdgeLoss,
                 supertagger: Supertagger,
                 lexlabeltagger: Supertagger,
                 supertagger_loss: SupertaggingLoss,
                 lexlabel_loss: SupertaggingLoss,
                 lexlabelcopier: Copier = None,
                 output_null_lex_label: bool = True,
                 loss_mixing: Dict[str, float] = None,
                 all_automaton_loss = False,
                 dropout: float = 0.0,
                 validation_evaluator: Optional[Evaluator] = None,
                 regularizer: Optional[RegularizerApplicator] = None):

        super().__init__(vocab, regularizer)
        self.name = name
        self.edge_model = edge_model
        self.supertagger = supertagger
        self.lexlabeltagger = lexlabeltagger
        self.supertagger_loss = supertagger_loss
        self.lexlabel_loss = lexlabel_loss
        self.loss_function = loss_function
        self.lexlabelcopier = lexlabelcopier
        self.loss_mixing = loss_mixing or dict()
        self.all_automaton_loss = all_automaton_loss
        self.validation_evaluator = validation_evaluator
        self.output_null_lex_label = output_null_lex_label

        self._dropout = InputVariationalDropout(dropout)

        for loss_name in AMAutomataTask.loss_names:
            if loss_name not in self.loss_mixing:
                self.loss_mixing[loss_name] = 1.0
                logger.info(f"Loss name {loss_name} not found in loss_mixing, using a weight of 1.0")
            else:
                if self.loss_mixing[loss_name] is None:
                    if loss_name not in ["supertagging", "lexlabel"]:
                        raise ConfigurationError(
                            "Only the loss mixing coefficients for supertagging and lexlabel may be None, but not " + loss_name)

        not_contained = set(self.loss_mixing.keys()) - set(AMAutomataTask.loss_names)
        if len(not_contained):
            logger.critical(f"The following loss name(s) are unknown: {not_contained}")
            raise ValueError(f"The following loss name(s) are unknown: {not_contained}")

        if self.all_automaton_loss:
            self.loss_mixing["edge_existence"] = 0.0
            self.loss_mixing["lexlabel"] = 0.0

        self._lexlabel_acc = CategoricalAccuracy()
        self._attachment_scores = AttachmentScores()
        self._running_amconll = []  # list of AmConllSentence
        self._inside_metric = Average()
        self._loss_metric = Average()
        self.current_epoch = 0

        tags = self.vocab.get_token_to_index_vocabulary("pos")
        punctuation_tag_indices = {tag: index for tag, index in tags.items() if tag in POS_TO_IGNORE}
        self._pos_to_ignore = set(punctuation_tag_indices.values())
        logger.info(f"Found POS tags corresponding to the following punctuation : {punctuation_tag_indices}. "
                    "Ignoring words with these POS tags for evaluation.")

        self.compute_softmax_for_scores = False  # set to true when dumping scores to incorporate softmax computation into computation time

    def check_all_dimensions_match(self, encoder_output_dim):

        check_dimensions_match(encoder_output_dim, self.edge_model.encoder_dim(),
                               "encoder output dim", self.name + " input dim edge model")
        check_dimensions_match(encoder_output_dim, self.supertagger.encoder_dim(),
                               "encoder output dim", self.name + " supertagger input dim")
        check_dimensions_match(encoder_output_dim, self.lexlabeltagger.encoder_dim(),
                               "encoder output dim", self.name + " lexical label tagger input dim")

    @overrides
    def forward(self,  # type: ignore
                encoded_text_parsing: torch.Tensor,
                encoded_text_tagging: torch.Tensor,
                mask: torch.Tensor,
                pos_tags: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                rule_index: torch.LongTensor = None,
                rule_mask: torch.LongTensor = None,
                lexlabels: torch.LongTensor = None,
                head_indices: torch.LongTensor = None,
                lemma_copying: torch.LongTensor = None,
                token_copying: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        Takes a batch of encoded sentences and returns a dictionary with loss and predictions.

        :param encoded_text_parsing: sentence representation of shape (batch_size, seq_len, encoder_output_dim)
        :param encoded_text_tagging: sentence representation of shape (batch_size, seq_len, encoder_output_dim)
            or None if formalism of batch doesn't need supertagging
        :param mask:  matching the sentence representation of shape (batch_size, seq_len)
        :param pos_tags: the accompanying pos tags (batch_size, seq_len)
        :param metadata:
        :param rule_index: for each rule (same order as in rule_iterator in metadata) what index of neural network
                            prediction corresponds to the rule (batch_size, rule_count)
        :param rule_mask: mask for rule_index
        :param lexlabels: the accompanying lexical labels (batch_size, seq_len)
        :param head_indices: the gold edge labels for each word (incoming edge, see amconll files) (batch_size, seq_len)
        :return:
        """

        print_time = False
        print_diagnostics = False  # = self.training

        start_time = time()
        encoded_text_parsing = self._dropout(encoded_text_parsing)
        if encoded_text_tagging is not None:
            encoded_text_tagging = self._dropout(encoded_text_tagging)

        batch_size, seq_len, _ = encoded_text_parsing.shape

        edge_existence_scores = self.edge_model.edge_existence(encoded_text_parsing,
                                                               mask)  # shape (batch_size, seq_len, seq_len)
        # print(f"edge_existence_scores: {edge_existence_scores}")

        # shape (batch_size, seq_len, num_supertags)
        if encoded_text_tagging is not None and self.supertagger is not None and self.lexlabeltagger is not None \
                and self.loss_mixing["supertagging"] is not None \
                and self.loss_mixing["lexlabel"] is not None:
            supertagger_logits = self.supertagger.compute_logits(encoded_text_tagging)
            lexlabel_logits = self.lexlabeltagger.compute_logits(
                encoded_text_tagging)  # shape (batch_size, seq_len, num label tags)
        else:
            supertagger_logits = None
            lexlabel_logits = None

        # Make predictions on data:
        if self.training:
            predicted_heads = self._greedy_decode_arcs(edge_existence_scores, mask)
            edge_label_logits = self.edge_model.label_scores(encoded_text_parsing,
                                                             predicted_heads)  # shape (batch_size, seq_len, num edge labels)
            predicted_edge_labels = self._greedy_decode_edge_labels(edge_label_logits)
        else:
            # Find best tree with CLE
            predicted_heads = cle_decode(edge_existence_scores, mask.data.sum(dim=1).long())
            # With info about tree structure, get edge label scores
            edge_label_logits = self.edge_model.label_scores(encoded_text_parsing, predicted_heads)
            # Predict edge labels
            predicted_edge_labels = self._greedy_decode_edge_labels(edge_label_logits)

        output_dict = {
            "heads": predicted_heads,
            "edge_existence_scores": edge_existence_scores,
            "label_logits": edge_label_logits,  # shape (batch_size, seq_len, num edge labels)
            "full_label_logits": self.edge_model.full_label_scores(encoded_text_parsing),
            # these are mostly required for the projective decoder
            "mask": mask,
            "words": [meta["words"] for meta in metadata],
            "attributes": [meta["attributes"] for meta in metadata],
            "token_ranges": [meta["token_ranges"] for meta in metadata],
            "encoded_text_parsing": encoded_text_parsing,
            "encoded_text_tagging": encoded_text_tagging,
            "position_in_corpus": [meta["position_in_corpus"] for meta in metadata],
            "formalism": self.name
        }

        if encoded_text_tagging is not None and self.loss_mixing["supertagging"] is not None:
            output_dict["supertag_scores"] = supertagger_logits  # shape (batch_size, seq_len, num supertags)
            output_dict["best_supertags"] = Supertagger.top_k_supertags(supertagger_logits, 1).squeeze(
                2)  # shape (batch_size, seq_len)

        if encoded_text_tagging is not None and self.loss_mixing["lexlabel"] is not None:
            if not self.output_null_lex_label:
                bottom_lex_label_index = self.vocab.get_token_index("_", namespace=self.name + "_lex_labels")
                masked_lexlabel_logits = lexlabel_logits.clone().detach()  # shape (batch_size, seq_len, num label tags)
                masked_lexlabel_logits[:, :, bottom_lex_label_index] = - 1e20
            else:
                masked_lexlabel_logits = lexlabel_logits

            output_dict["lexlabels"] = Supertagger.top_k_supertags(masked_lexlabel_logits, 1).squeeze(
                2)  # shape (batch_size, seq_len)

        is_annotated = metadata[0]["is_annotated"]
        if any(metadata[i]["is_annotated"] != is_annotated for i in range(batch_size)):
            raise ValueError("Batch contained inconsistent information if data is annotated.")
        # if "is_inherently_annotated" in metadata[0]:
        #     if any(metadata[i]["is_inherently_annotated"] != metadata[0]["is_inherently_annotated"] for i in
        #            range(batch_size)):
        #         print("Batch contained inconsistent information if data is annotated.")
        #         for meta in metadata:
        #             if "is_inherently_annotated" in meta and not meta["is_inherently_annotated"]:
        #                 print('meta["is_inherently_annotated"]' + str(meta["is_inherently_annotated"]))
        #                 r = []
        #                 for i, w in enumerate(meta["words"]):
        #                     fields = list(w)
        #                     if fields[-1] is None:
        #                         fields = fields[:-1]  # when token range not present -> remove it
        #                     r.append("\t".join([str(x) for x in [i] + fields]))
        #                 print("\n".join(r) + "\n")

        orig_lexlabel_logits = lexlabel_logits # TODO I think this is not compatible with copying at this point
        lexlabel_logits = lexlabel_logits[:, 1:, :].contiguous()
        if self.lexlabelcopier is not None:
            lexlabel_prob_matrix, p_vocab, p_lemma, p_token = self.lexlabelcopier.compute_logprobs(
                encoded_text_tagging[:, 1:, :].contiguous(),
                lemma_copying,
                token_copying,
                lexlabels,
                lexlabel_logits)
            # TODO lexlabel_logprob_matrix is none if lemma_copying or token_copying are None, which should be
            # TODO exactly if (not is_annotated). Then we also won't need lexlabel_logprob_matrix. Could be cleaner. -- JG Jan 21
            # print("diagonal")
            # print(diagonal)
            # diagonal.register_hook(lambda grad: print(f"diagonal grad: {grad}"))
            # lexlabel_nll.register_hook(lambda grad: print(f"lexlabel_nll grad: {grad}"))
            output_dict["p_vocab"] = p_vocab  # shape is batch_size x (seq_length-1) x 1
            output_dict["p_lemma"] = p_lemma
            output_dict["p_token"] = p_token

        # Compute loss:
        if is_annotated and head_indices is not None and rule_index is not None:
            #  torch.autograd.set_detect_anomaly(True)  # this allows pinpoining errors better in debugging, but costs speed

            # convert neural outputs into a format (view) that makes it possible to associate them with automaton rules via rule_index
            # To be precise, the indices stored in rule_index[i] correspond to indices of all_logprobs[i] (i = entry in batch)
            st_size = supertagger_logits.size()
            el_size = edge_label_logits.size()
            if print_diagnostics:
                print(f"st_size: {st_size}")
                print(f"el_size: {el_size}")
                print(f"orig_lexlabel_logits size: {orig_lexlabel_logits.size()}")
                print(f"lexlabels size: {lexlabels.size()}")
                print(f"head_indices size: {head_indices.size()}")
                print(f"edge_existence_scores size: {edge_existence_scores.size()}")

            supertagger_logprobs = torch.nn.functional.log_softmax(supertagger_logits, dim=2)
            edge_label_logprobs = torch.nn.functional.log_softmax(edge_label_logits, dim=2)
            if self.all_automaton_loss:
                lexlabel_logprobs = torch.nn.functional.log_softmax(orig_lexlabel_logits, dim=2)
                edge_existence_logprobs = torch.nn.functional.log_softmax(edge_existence_scores, dim=2)
                cuda_check = lexlabels.is_cuda
                if cuda_check:
                    device = lexlabels.get_device()
                else:
                    device = "cpu"
                lexlabel_buffer = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                # print(lexlabel_buffer)
                lexlabel_lookup = torch.cat([lexlabel_buffer, lexlabels], dim=1).view(batch_size, seq_len, 1)
                # print(lexlabel_lookup)
                # print(lexlabel_logits)
                lexlabel_logprobs = torch.gather(lexlabel_logprobs, 2, lexlabel_lookup)
                edge_existence_logprobs = torch.gather(edge_existence_logprobs, 2,
                                                       head_indices.view(batch_size, seq_len, 1))

                tag_logprobs = supertagger_logprobs + lexlabel_logprobs  # with broadcasting
                edge_logprobs = edge_label_logprobs + edge_existence_logprobs  # with broadcasting
                if print_diagnostics:
                    print(f"supertagger_logprobs.size {supertagger_logprobs.size()}")
                    print(f"lexlabel_logprobs.size {lexlabel_logprobs.size()}")
                    print(f"tag_logprobs.size {tag_logprobs.size()}")
                    print(f"edge_label_logprobs.size {edge_label_logprobs.size()}")
                    print(f"edge_existence_logprobs.size {edge_existence_logprobs.size()}")
                    print(f"edge_logprobs.size {edge_logprobs.size()}")
                all_logprobs = torch.cat(
                    [tag_logprobs.view(batch_size, -1), edge_logprobs.view(batch_size, -1)],
                    dim=1)  # dimension batch_len x (sent_len * (st_label_count + edge_label_count))
            else:
                all_logprobs = torch.cat(
                    [supertagger_logprobs.view(st_size[0], -1), edge_label_logprobs.view(el_size[0], -1)],
                    dim=1)  # dimension batch_len x (sent_len * (st_label_count + edge_label_count))

            # printing some information for testing
            # print(f"all_logprobs.size(): {all_logprobs.size()}")
            # print(f"rule_index.size(): {rule_index.size()}")
            # print(f"rule_index: {rule_index}")
            # rule_iterator = metadata[0]["rule_iterator"]
            # automaton = metadata[0]["automaton"]
            # for rule, logit_index in zip(to_python(rule_iterator), rule_index[0]):
            #     print(rule.toString(automaton))
            #     print(logit_index)
            #     print(all_logprobs[0][logit_index])

            if print_time:
                print(f"time before automaton loss: {time() - start_time}")
            start_time = time()
            # automaton loss
            if print_diagnostics:
                print(f"rule_index {rule_index}")
            logprobs_for_rules_premask = torch.gather(all_logprobs, 1, rule_index)
            logprobs_for_rules = logprobs_for_rules_premask * rule_mask.int().float()
            if print_diagnostics:
                print(f"rule_mask: {rule_mask}")
                # print(f"logprobs_for_rules[0]: {logprobs_for_rules[0]}")

            if print_time:
                print(f"time for gathering and masking: {time() - start_time}")
            start_time = time()
            outer_weights_python = []
            # iterate over each entry in batch:
            for logits, meta, indices in zip(logprobs_for_rules, metadata, rule_index):
                logits_python = logits.tolist()  # need to convert to python primitive to send it to java as a float[]
                all_rules_in_bottom_up_order = meta["all_rules_in_bottom_up_order"]
                max_state_id_plus_one = meta["max_state_id_plus_one"]
                final_states = meta["final_states"]
                outer_weights_python.append(PyjniusHelper.computeOuterProbabilities(logits_python,
                                                                                    all_rules_in_bottom_up_order,
                                                                                    max_state_id_plus_one,
                                                                                    final_states))
                # get total inside for metrics
                total_inside = PyjniusHelper.getTotalLogInside(logits_python, all_rules_in_bottom_up_order,
                                                               max_state_id_plus_one, final_states)
                self._inside_metric(total_inside)
                if print_diagnostics:
                    print(f"log inside: {total_inside}")
                    print("indices:")
                    for index in indices:
                        print(index.item())
                    print("rule weights:")
                    for logit in logits:
                        print(logit.item())
                    print("automaton:")
                    # the following are no longer possible since we don't carry the automaton around anymore.
                    # If you need to print them, add automaton to meta again in amconll_automata DatasetReader
                    # for rule in to_python(all_rules_in_bottom_up_order):
                    #     print(rule.toString(automaton))
                    # print(f"viterbi: {automaton.viterbi().toString()}")
                    print("outer weights:")
                for i, (outer_weight, rule_weight) in enumerate(zip(outer_weights_python[-1], logits_python)):
                    if abs(outer_weight) < 0.000001:  # TODO slight hack to get the non-automaton outer weights
                        outer_weight = -rule_weight  # when seeing this as rule of auto that is obligatory, corresponds to outer/inside
                        outer_weights_python[-1][i] = outer_weight
                    if print_diagnostics:
                        print(outer_weight)

                viterbi_rule_list_and_weight = PyjniusHelper.getViterbi(logits_python, all_rules_in_bottom_up_order,
                                                                 max_state_id_plus_one, final_states)
                amconll_attributes = dict(meta["attributes"])
                amconll_attributes["predicted_weight"] = str(viterbi_rule_list_and_weight.right)
                amconll_attributes["total_inside_weight"] = str(numpy.exp(total_inside))
                amconll_sent = AMSentence(meta["words"], amconll_attributes)
                outgoing_edge_positions = set()
                for rule in viterbi_rule_list_and_weight.left:
                    if meta["supertag_map"].containsKey(rule):
                        position_and_supertag = meta["supertag_map"].get(rule)
                        st_and_type = AMSentence.split_supertag(position_and_supertag.right)
                        word = amconll_sent.words[position_and_supertag.left-1]
                        word = word.set_fragment(st_and_type[0])
                        word = word.set_typ(st_and_type[1])
                        amconll_sent.words[position_and_supertag.left-1] = word
                    elif meta["edge_map"].containsKey(rule):
                        positions_and_edgelabel = meta["edge_map"].get(rule)
                        word = amconll_sent.words[positions_and_edgelabel.left.right-1]
                        #  word = word.set_head(positions_and_edgelabel.left.left)  # not actually necessary, already in the word
                        outgoing_edge_positions.add(positions_and_edgelabel.left.left-1)
                        word = word.set_edge_label(positions_and_edgelabel.right)
                        amconll_sent.words[positions_and_edgelabel.left.right-1] = word
                for i, word in enumerate(amconll_sent.words):
                    if word.head == 0:
                        if i in outgoing_edge_positions:
                            word = word.set_edge_label("ROOT")
                        else:
                            word = word.set_edge_label("IGNORE")
                    amconll_sent.words[i] = word
                self._running_amconll.append(amconll_sent)

            if print_time:
                print(f"time for inside outside: {time() - start_time}")
            start_time = time()

            # print(outer_weights_python)

            # back to pytorch tensors
            outer_weights = logprobs_for_rules.new(outer_weights_python)
            # print(f"outer_weights[0]: {outer_weights[0]}")

            # compute loss batched, returning a vector with a loss for each entry in the batch
            # print(outer_weights+logprobs_for_rules)
            # print(torch.logsumexp(outer_weights+logprobs_for_rules, dim=1))
            inner_sum_vector = outer_weights + logprobs_for_rules
            # batch_loss_tensor = -torch.sum(inner_sum_vector, dim=1)
            batch_loss_tensor = -torch.logsumexp(inner_sum_vector, dim=1)  # this is too much rich get richer
            # print(batch_loss_tensor)
            # sum the loss over all entries in the batch
            loss = torch.sum(batch_loss_tensor)

            if print_diagnostics:
                batch_loss_tensor.register_hook(lambda grad: print(f"batch_loss_tensor grad: {grad}"))
                inner_sum_vector.register_hook(lambda grad: print(f"inner_sum_vector grad: {grad}"))
                logprobs_for_rules.register_hook(lambda grad: print(f"logprobs_for_rules grad: {grad}"))
                logprobs_for_rules_premask.register_hook(lambda grad: print(f"logprobs_for_rules_premask grad: {grad}"))
                edge_label_logits.register_hook(lambda grad: print(f"edge_label_logits grad: {grad}"))

                print(f"Tree inside loss: {loss}")
            loss = mix_loss(self.loss_mixing["supertagging"], loss)

            if print_time:
                print(f"time for rest of automaton loss: {time() - start_time}")
            start_time = time()

            # gold_edge_label_logits = self.edge_model.label_scores(encoded_text_parsing, head_indices)
            # edge_label_loss = self.loss_function.label_loss(gold_edge_label_logits, mask, head_tags)
            if self.loss_mixing["edge_existence"] > 0:
                edge_existence_loss = self.loss_function.edge_existence_loss(edge_existence_scores, head_indices, mask)
            else:
                edge_existence_loss = 0
            if print_diagnostics:
                print(f"edge_existence loss: {edge_existence_loss}")

            # compute loss, remove loss for artificial root
            # if encoded_text_tagging is not None and self.loss_mixing["supertagging"] is not None:
            #     supertagger_logits = supertagger_logits[:, 1:, :].contiguous()
            #     supertagging_nll = self.supertagger_loss.loss(supertagger_logits, supertags, mask[:, 1:])
            # else:
            #     supertagging_nll = None

            if encoded_text_tagging is not None and self.loss_mixing["lexlabel"] is not None and self.loss_mixing["lexlabel"] > 0:
                if self.lexlabelcopier is not None:
                    lexlabel_logprob_diagonal = torch.log(torch.diagonal(lexlabel_prob_matrix, dim1=1, dim2=2))
                    lexlabel_nll = -torch.sum(lexlabel_logprob_diagonal)  # negative log likelihood loss
                else:
                    lexlabel_nll = self.lexlabel_loss.loss(lexlabel_logits, lexlabels, mask[:, 1:])
            else:
                lexlabel_nll = None

            loss += mix_loss(self.loss_mixing["edge_existence"],
                             edge_existence_loss)  # +  mix_loss(self.loss_mixing["edge_label"], edge_label_loss)

            # if supertagging_nll is not None:
            #     loss += mix_loss(self.loss_mixing["supertagging"], supertagging_nll)
            if lexlabel_nll is not None:
                lexlabel_loss = mix_loss(self.loss_mixing["lexlabel"], lexlabel_nll)
                if print_diagnostics:
                    print(f"lexlabel loss: {lexlabel_loss}")
                loss += lexlabel_loss

            # Compute LAS/UAS/Supertagging acc/Lex label acc:
            evaluation_mask = self._get_mask_for_eval(mask[:, 1:], pos_tags)
            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            if edge_existence_loss is not None:  # and edge_label_loss is not None:
                self._attachment_scores(predicted_heads[:, 1:],
                                        predicted_edge_labels[:, 1:],
                                        head_indices[:, 1:],
                                        predicted_edge_labels[:, 1:],
                                        # this should be gold, but this way we still get UAS
                                        evaluation_mask)
            evaluation_mask = mask[:, 1:].contiguous()
            # if supertagging_nll is not None:
            #     self._top_6supertagging_acc(supertagger_logits, supertags, evaluation_mask)
            #     self._supertagging_acc(supertagger_logits, supertags, evaluation_mask)  # compare against gold data
            if lexlabel_nll is not None:
                self._lexlabel_acc(lexlabel_logits, lexlabels, evaluation_mask)  # compare against gold data

            output_dict["arc_loss"] = edge_existence_loss
            # output_dict["edge_label_loss"] = edge_label_loss
            # output_dict["supertagging_loss"] = supertagging_nll
            output_dict["lexlabel_loss"] = lexlabel_nll
            output_dict["loss"] = loss
            self._loss_metric(loss.data.item())

        if self.compute_softmax_for_scores:
            # We don't use the results but we want it to be included in the time measurement
            # See dump_scores what part of computation is done outside of the time measurement in forward()
            F.log_softmax(output_dict["full_label_logits"], 3)
            F.log_softmax(output_dict["supertag_scores"], 2)
            torch.argsort(output_dict["supertag_scores"], descending=True, dim=2)

        if print_time:
            print(f"time after automaton loss: {time() - start_time}")

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        """
        In contrast to its name, this function does not perform the decoding but only prepares it.
        :param output_dict:
        :return:
        """
        if self.supertagger is not None and self.loss_mixing[
            "supertagging"] is not None:  # we have a supertagger, so this is proper AM dependency parsing
            return self.prepare_for_ftd(output_dict)
        else:  # we don't have a supertagger, perform good old dependency parsing
            return self.only_cle(output_dict)

    def only_cle(self, output_dict: Dict[str, torch.Tensor]):
        """
        Therefore, we take the result of forward and perform the following steps (for each sentence in batch):
        - remove padding
        :param output_dict: result of forward
        :return: output_dict with the following keys added:
            - lexlabels: nested list: contains for each sentence, for each word the most likely lexical label (w/o artificial root)
            - supertags: nested list: contains for each sentence, for each word the most likely lexical label (w/o artificial root)
        """
        full_label_logits = output_dict.pop(
            "full_label_logits").cpu().detach().numpy()  # shape (batch size, seq len, seq len, num edge labels)
        edge_existence_scores = output_dict.pop(
            "edge_existence_scores").cpu().detach().numpy()  # shape (batch size, seq len, seq len, num edge labels)
        heads = output_dict.pop("heads")
        heads_cpu = heads.cpu().detach().numpy()
        mask = output_dict.pop("mask")
        edge_label_logits = output_dict.pop(
            "label_logits").cpu().detach().numpy()  # shape (batch_size, seq_len, num edge labels)

        output_dict.pop("encoded_text_parsing")
        output_dict.pop("encoded_text_tagging")  # don't need that

        lengths = get_lengths_from_binary_sequence_mask(mask)

        # here we collect things, in the end we will have one entry for each sentence:
        all_edge_label_logits = []
        head_indices = []
        all_full_label_logits = []
        all_edge_existence_scores = []

        for i, length in enumerate(lengths):
            instance_heads_cpu = list(heads_cpu[i, 1:length])
            # apply changes to instance_heads tensor:
            instance_heads = heads[i, :]
            for j, x in enumerate(instance_heads_cpu):
                instance_heads[j + 1] = torch.tensor(
                    x)  # +1 because we removed the first position from instance_heads_cpu

            all_edge_label_logits.append(edge_label_logits[i, 1:length, :])

            all_full_label_logits.append(full_label_logits[i, :length, :length, :])
            all_edge_existence_scores.append(edge_existence_scores[i, :length, :length])
            head_indices.append(instance_heads_cpu)

        output_dict["label_logits"] = all_edge_label_logits
        output_dict["predicted_heads"] = head_indices
        output_dict["full_label_logits"] = all_full_label_logits
        output_dict["edge_existence_scores"] = all_edge_existence_scores
        return output_dict

    def prepare_for_ftd(self, output_dict: Dict[str, torch.Tensor]):
        """
        This function does not perform the decoding but only prepares it.
        Therefore, we take the result of forward and perform the following steps (for each sentence in batch):
        - remove padding
        - identify the root of the sentence, group other root-candidates under the proper root
        - collect a selection of supertags to speed up computation (top k selection is done later)
        :param output_dict: result of forward
        :return: output_dict with the following keys added:
            - lexlabels: nested list: contains for each sentence, for each word the most likely lexical label (w/o artificial root)
            - supertags: nested list: contains for each sentence, for each word the most likely lexical label (w/o artificial root)
        """
        t0 = time()
        best_supertags = output_dict.pop("best_supertags").cpu().detach().numpy()
        supertag_scores = output_dict.pop("supertag_scores")  # shape (batch_size, seq_len, num supertags)
        full_label_logits = output_dict.pop(
            "full_label_logits").cpu().detach().numpy()  # shape (batch size, seq len, seq len, num edge labels)
        edge_existence_scores = output_dict.pop(
            "edge_existence_scores").cpu().detach().numpy()  # shape (batch size, seq len, seq len, num edge labels)
        k = 10
        if self.validation_evaluator:  # retrieve k supertags from validation evaluator.
            if isinstance(self.validation_evaluator.predictor, AMconllPredictor):
                k = self.validation_evaluator.predictor.k
        k += 10  # perhaps there are some ill-formed supertags, make that very unlikely that there are not enough left after filtering.
        top_k_supertags = Supertagger.top_k_supertags(supertag_scores,
                                                      k).cpu().detach().numpy()  # shape (batch_size, seq_len, k)
        supertag_scores = supertag_scores.cpu().detach().numpy()
        lexlabels = output_dict.pop("lexlabels").cpu().detach().numpy()  # shape (batch_size, seq_len)
        if self.lexlabelcopier is not None:
            p_vocab = output_dict.pop("p_vocab").cpu().detach().numpy()  # shape (batch_size, seq_len-1, 1)
            p_lemma = output_dict.pop("p_lemma").cpu().detach().numpy()  # shape (batch_size, seq_len-1, 1)
            p_token = output_dict.pop("p_token").cpu().detach().numpy()  # shape (batch_size, seq_len-1, 1)
            words = output_dict["words"]
        heads = output_dict.pop("heads")
        heads_cpu = heads.cpu().detach().numpy()
        mask = output_dict.pop("mask")
        edge_label_logits = output_dict.pop(
            "label_logits").cpu().detach().numpy()  # shape (batch_size, seq_len, num edge labels)
        encoded_text_parsing = output_dict.pop("encoded_text_parsing")
        output_dict.pop("encoded_text_tagging")  # don't need that
        lengths = get_lengths_from_binary_sequence_mask(mask)

        # here we collect things, in the end we will have one entry for each sentence:
        all_edge_label_logits = []
        all_supertags = []
        head_indices = []
        roots = []
        all_predicted_lex_labels = []
        all_full_label_logits = []
        all_edge_existence_scores = []
        all_supertag_scores = []

        # we need the following to identify the root
        root_edge_label_id = self.vocab.get_token_index("ROOT", namespace=self.name + "_head_tags")
        bot_id = self.vocab.get_token_index(AMSentence.get_bottom_supertag(), namespace=self.name + "_supertag_labels")

        for i, length in enumerate(lengths):
            instance_heads_cpu = list(heads_cpu[i, 1:length])
            # Postprocess heads and find root of sentence:
            instance_heads_cpu, root = find_root(instance_heads_cpu, best_supertags[i, 1:length],
                                                 edge_label_logits[i, 1:length, :], root_edge_label_id, bot_id,
                                                 modify=True)
            roots.append(root)
            # apply changes to instance_heads tensor:
            instance_heads = heads[i, :]
            for j, x in enumerate(instance_heads_cpu):
                instance_heads[j + 1] = torch.tensor(
                    x)  # +1 because we removed the first position from instance_heads_cpu

            # re-calculate edge label logits since heads might have changed:
            label_logits = self.edge_model.label_scores(encoded_text_parsing[i].unsqueeze(0),
                                                        instance_heads.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            # (un)squeeze: fake batch dimension
            all_edge_label_logits.append(label_logits[1:length, :])

            all_full_label_logits.append(full_label_logits[i, :length, :length, :])
            all_edge_existence_scores.append(edge_existence_scores[i, :length, :length])

            # calculate supertags for this sentence:
            all_supertag_scores.append(supertag_scores[i, 1:length, :])  # new shape (sent length, num supertags)
            supertags_for_this_sentence = []
            for word in range(1, length):
                supertags_for_this_word = []
                for top_k in top_k_supertags[i, word]:
                    fragment, typ = AMSentence.split_supertag(
                        self.vocab.get_token_from_index(top_k, namespace=self.name + "_supertag_labels"))
                    score = supertag_scores[i, word, top_k]
                    supertags_for_this_word.append((score, fragment, typ))
                if bot_id not in top_k_supertags[
                    i, word]:  # \bot is not in the top k, but we have to add it anyway in order for the decoder to work properly.
                    fragment, typ = AMSentence.split_supertag(AMSentence.get_bottom_supertag())
                    supertags_for_this_word.append((supertag_scores[i, word, bot_id], fragment, typ))
                supertags_for_this_sentence.append(supertags_for_this_word)
            all_supertags.append(supertags_for_this_sentence)
            if self.lexlabelcopier is not None:
                lexlabels_decoded = []
                for j in range(1, length):
                    lemma = words[i][j - 1].lemma
                    token = words[i][j - 1].token
                    label_id = lexlabels[i][j]
                    probability_triple = numpy.array([p_vocab[i][j - 1, 0], p_lemma[i][j - 1, 0], p_token[i][j - 1, 0]])
                    lexlabels_decoded.append(
                        self.lexlabelcopier.get_most_likely(probability_triple, label_id, self.vocab,
                                                            self.name + "_lex_labels", lemma, token))
                # print("lexlabels_decoded")
                # print(lexlabels_decoded)
                all_predicted_lex_labels.append(lexlabels_decoded)
            else:
                all_predicted_lex_labels.append(
                    [self.vocab.get_token_from_index(label, namespace=self.name + "_lex_labels") for label in
                     lexlabels[i, 1:length]])
            head_indices.append(instance_heads_cpu)

        t1 = time()
        normalized_diff = (t1 - t0) / len(lengths)
        output_dict["normalized_prepare_ftd_time"] = [normalized_diff for _ in range(len(lengths))]
        output_dict["lexlabels"] = all_predicted_lex_labels
        output_dict["supertags"] = all_supertags
        output_dict["root"] = roots
        output_dict["label_logits"] = all_edge_label_logits
        output_dict["predicted_heads"] = head_indices
        output_dict["full_label_logits"] = all_full_label_logits
        output_dict["edge_existence_scores"] = all_edge_existence_scores
        output_dict["supertag_scores"] = all_supertag_scores
        return output_dict

    def _greedy_decode_edge_labels(self, edge_label_logits: torch.Tensor) -> torch.Tensor:
        """
        Assigns edge labels according to (existing) edges.
        Parameters
        ----------
        edge_label_logits: ``torch.Tensor`` of shape (batch_size, sequence_length, num_head_tags)

        Returns
        -------
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded head tags (labels of incoming edges) of each word.
        """
        _, head_tags = edge_label_logits.max(dim=2)
        return head_tags

    def _greedy_decode_arcs(self,
                            existence_scores: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """
        Decodes the head  predictions by decoding the unlabeled arcs
        independently for each word. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        existence_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        mask: torch.Tensor, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        existence_scores = existence_scores + torch.diag(existence_scores.new(mask.size(1)).fill_(-numpy.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).byte().unsqueeze(2)
            # existence_scores = existence_scores.masked_fill(minus_mask, -numpy.inf)
            # use inplace operation to save memory (note the underscore at the end of the operation)
            existence_scores.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = existence_scores.max(dim=2)
        return heads

    def _get_mask_for_eval(self,
                           mask: torch.LongTensor,
                           pos_tags: torch.LongTensor) -> torch.LongTensor:
        """
        Dependency evaluation excludes words are punctuation.
        Here, we create a new mask to exclude word indices which
        have a "punctuation-like" part of speech tag.

        Parameters
        ----------
        mask : ``torch.LongTensor``, required.
            The original mask.
        pos_tags : ``torch.LongTensor``, required.
            The pos tags for the sequence.

        Returns
        -------
        A new mask, where any indices equal to labels
        we should be ignoring are masked.
        """
        new_mask = mask.detach()
        for label in self._pos_to_ignore:
            label_mask = pos_tags.eq(label).long()
            new_mask = new_mask * (1 - label_mask)
        return new_mask

    def metrics(self, parser_model, reset: bool = False, model_path=None) -> Dict[str, float]:
        """
        Is called by a GraphDependencyParser
        :param parser_model: a GraphDependencyParser
        :param reset:
        :return:
        """
        r = self.get_metrics(reset)
        if reset:  # epoch done
            if self.training:  # done on the training data
                self.current_epoch += 1
            else:  # done on dev/test data
                if self.validation_evaluator:
                    metrics = self.validation_evaluator.eval(parser_model, self.current_epoch, model_path)
                    for name, val in metrics.items():
                        r[name] = val
        return r

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        r = self._attachment_scores.get_metric(reset)
        # if self.loss_mixing["supertagging"] is not None:
        #     r["Constant_Acc"] = self._supertagging_acc.get_metric(reset)
        #     r["Constant_Acc_6_best"] = self._top_6supertagging_acc.get_metric(reset)
        if self.loss_mixing["lexlabel"] is not None:
            r["Label_Acc"] = self._lexlabel_acc.get_metric(reset)
        r["Average_Insides"] = self._inside_metric.get_metric(reset)
        r["Loss_Minus_Insides"] = self._loss_metric.get_metric(reset) - r["Average_Insides"]
        # las = r["LAS"]
        # if "Constant_Acc" in r:
        #     r["mean_constant_acc_las"] = (las + r["Constant_Acc"]) / 2
        r["amconll_list"] = self._running_amconll
        if reset:
            self._running_amconll = []
        return r
