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
import time
from typing import Dict, Optional, Any, List
import logging

from overrides import overrides
import numpy
import torch
import torch.nn as nn

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, F1Measure, SequenceAccuracy

# from graph_dependency_parser.components.spacy_token_embedder import TokenToVec


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# note: took inspiration form graph_dependency_parser, allennlp's nla_semparse
# and Joe Barrow's allenNLP tutorial
@Model.register("typetaggingmodel")
class TypeTaggingModel(Model):
    """
    Predict AM types from tokens and AM types for another formalism

    ...
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 src_type_embedding: Embedding,
                 pos_tag_embedding: Embedding = None,
                 encoder: Seq2SeqEncoder = Seq2SeqEncoder,
                 # decoder
                 ) -> None:
        super().__init__(vocab)
        self._embedder = text_field_embedder
        # print(self._embedder)  # debug print
        self._pos_tag_embedding = pos_tag_embedding
        self._src_type_embedding = src_type_embedding
        self._encoder = encoder
        self.label_namespace = 'tgt_types'
        self.num_classes = vocab.get_vocab_size(self.label_namespace)

        # input dimension: words + src_types  + postags
        representation_dim = text_field_embedder.get_output_dim()
        representation_dim += src_type_embedding.get_output_dim()
        if pos_tag_embedding is not None:
           representation_dim += pos_tag_embedding.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        # todo: copied classifer from Joe Barrow's tutorial: change?
        self._classifier = nn.Linear(in_features=encoder.get_output_dim(),
                                    out_features=self.num_classes)
        # todo: delete debugging messages
        print("DEBUG: Vocab size of target types: ", self.num_classes)
        print("DEBUG: Vocab: tok2index for tgt types: ", vocab.get_index_to_token_vocabulary(self.label_namespace))

        # todo: which metric(s)?
        # SpanBasedF1Measure(vocab, 'tgt_types'), SequenceAccuracy()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3),
        }
        return

    # note: param names must match field names (see text_to_instance )
    @overrides
    def forward(self,
                src_words: Dict[str, torch.LongTensor],
                src_pos: torch.LongTensor,
                src_types: torch.LongTensor,
                #lemmas: torch.LongTensor,
                #ner_tags: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                tgt_types: torch.LongTensor = None
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network (optionally computing loss)

        :param src_words:
        :param src_pos: Part-of-speech tags for the words. shape: `(batch_size, seq_len)`.
        :param src_types: Graph constant types on the source side. shape: `(batch_size, seq_len)`.
        :param metadata: some metadata info (e.g. sentence ids, the words...)
        :param tgt_types: `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class
            target types of shape `(batch_size, seq_len)`.
        :return: an output dictionary consisting of:
        - logits
            A tensor of shape `(batch_size, num_tokens, tgt_type_vocab_size)`
            representing unnormalised log probabilities of the tag classes.
        - class_probabilities
            A tensor of shape `(batch_size, num_tokens, tgt_type_vocab_size)`
            representing a distribution of the target types per word.
        - metadata: same metadata as input
        - mask:  info about padding in the batch (boolean tensor)
        - loss (only if tgt_types not None, e.g. when training the network)
        """
        # 1. Input consists of words, source types (and PoS tags?): concatenate
        concatenated_input = list()
        # print([entry.token for entry in metadata[0]['src_words']]) # debug
        # print(metadata[0]['src_attributes'])  # debug
        concatenated_input.append(self._embedder(src_words))
        #concatenated_input.append(self._embedder(src_types))
        concatenated_input.append(self._src_type_embedding(src_types))
        if src_pos is not None and self._pos_tag_embedding is not None:
            concatenated_input.append(self._pos_tag_embedding(src_pos))
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")
        #if len(concatenated_input) > 1:
        embedded_text_input = torch.cat(concatenated_input, -1)
        # Shape: (batch_size, seq_len, total_embedding_dim)

        # 1b: Input masking (padding in batch and such)
        mask = get_text_field_mask(src_words)
        # Shape: (batch_size, seq_len) value: bool (True=Unmasked)

        # 2. encode
        encoded = self._encoder(embedded_text_input, mask)
        # question: tok2vec used in graph_dependency_parser: needed here too?
        # Shape: (batch_size, seq_len, encoder_dim)

        # 3. decode/classifier
        type_logits = self._classifier(encoded)
        # Shape: (batch_size, seq_len, output_vocab_size)
        # todo: needed for make_output_human_readable ? -> move there?
        reshaped_log_probs = type_logits.view(-1, self.num_classes)
        batch_size, sequence_length, _ = embedded_text_input.size()
        class_probabilities = \
            torch.nn.functional.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
            )

        # 4. prepare output  # todo what return in output
        output: Dict[str, torch.Tensor] = {}
        output["mask"] = mask
        output["metadata"] = metadata
        output["logits"] = type_logits  # todo ok?
        # needed for make_output_human_readable:
        output["class_probabilities"] = class_probabilities

        if tgt_types is not None:  # evaluate against gold (=tgt_types)
            # tgt_types shape: (batch_size, seq_len) IDs of gold output types
            for name, metric in self.metrics.items():
                metric(type_logits, tgt_types, mask)

            # todo: change loss function?
            output["loss"] = sequence_cross_entropy_with_logits(type_logits,
                                                                tgt_types, mask)
        return output

    # https://docs.allennlp.org/main/api/models/model/#make_output_human_readable
    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        adds human readable output to the dict returned by forward()

        NOTE: Copied from allennlp simple_tagger.py (slightly changed)
        Does a simple position-wise argmax over each token,
        converts indices to string labels, and
        adds a `"predicted_types"` key to the dictionary with the result.
        :param output_dict : result of self.forward
        :return same dict as the input, but with additional key,value pair
        """
        # todo: is there an easier solution? (without numpy, ...)
        all_predictions = output_dict["class_probabilities"]
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in
                                range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace=self.label_namespace) for x in argmax_indices]
            all_tags.append(tags)
        output_dict["predicted_types"] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # return {'accuracy': 0.8, ...}
        return {metric_name: metric.get_metric(reset)
                for metric_name, metric in self.metrics.items()}
