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
import torch
import torch.nn as nn

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import F1Measure, SequenceAccuracy

# from graph_dependency_parser.components.spacy_token_embedder import TokenToVec


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# note: took inspiration form graph_dependency_parser, allennlp's nla_semparse
# and Joe Barrow's allenNLP tutorial
@Model.register("typeseq2seqmodel")
class TypeSeq2SeqModel(Model):
    """
    Predict AM types from tokens and AM types for another formalism

    ...
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 pos_tag_embedding: Embedding = None,
                 src_type_embedding: Embedding = None,  # todo: shouldn't be None
                 encoder: Seq2SeqEncoder = Seq2SeqEncoder,
                 # decoder
                 ) -> None:
        super().__init__(vocab)
        self._embedder = text_field_embedder
        print(self._embedder)  # debug print
        self._pos_tag_embedding = pos_tag_embedding
        self._src_type_embedding = src_type_embedding
        self._encoder = encoder

        # input dimension: words + src_types  + postags
        representation_dim = text_field_embedder.get_output_dim()
        representation_dim += src_type_embedding.get_output_dim()
        if pos_tag_embedding is not None:
           representation_dim += pos_tag_embedding.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        # todo: copied classifer from Joe Barrow's tutorial: change?
        self._classifier = nn.Linear(in_features=encoder.get_output_dim(),
                                    out_features=vocab.get_vocab_size('tgt_types'))

        # self._f1 = SpanBasedF1Measure(vocab, 'tgt_types')
        self._acc = SequenceAccuracy()
        pass

    # note: param names must match field names (see text_to_instance )
    # todo: need to debug forward function
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
        # 1. Input consists of words, source types (and PoS tags?): concatenate
        concatenated_input = list()
        #concatenated_input.append(self.text_field_embedder(src_words))
        print([entry.token for entry in metadata[0]['src_words']])  # debug print
        print([entry.typ for entry in metadata[0]['tgt_words']])  # debug print
        print(metadata[0]['src_attributes'])  # debug print
        print("Source words: ", src_words)  # debug print
        concatenated_input.append(self._embedder(src_words))
        #concatenated_input.append(self._embedder(src_types))
        concatenated_input.append(self._src_type_embedding(src_types))
        if src_pos is not None and self._pos_tag_embedding is not None:
            concatenated_input.append(self._pos_tag_embedding(src_pos))
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")
        #if len(concatenated_input) > 1:
        embedded_text_input = torch.cat(concatenated_input, -1)
        print("EmbdTI size: ", embedded_text_input.size())  # debug print
        # todo: masking?
        mask = get_text_field_mask(src_words)

        # 2. encode
        encoded = self._encoder(embedded_text_input, mask)
        # question: tok2vec used in graph_dependency_parser: needed here too?

        # 3. decode/classifier
        classified = self._classifier(encoded)
        print("Classified size: ", classified.size())  # debug print
        print("Tgt types size: ", tgt_types.size())  # debug print

        self._acc(classified, tgt_types, mask)  # todo: change metric?

        # 4. prepare output  # todo what return in output
        output: Dict[str, torch.Tensor] = {}
        output["mask"] = mask
        output["metadata"] = metadata
        output["results"] = classified  # todo ok?
        if tgt_types is not None:
            # todo: change loss function?
            output["loss"] = sequence_cross_entropy_with_logits(classified,
                                                                tgt_types, mask)
        return output

    # https://docs.allennlp.org/main/api/models/model/#make_output_human_readable
    # @overrides
    # def make_output_human_readable(
    #     self, output_dict: Dict[str, torch.Tensor]
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     todo write docstring
    #     :param output_dict: result of self.forward
    #     :return: same dict as method input but human readable
    #     """
    #     print("to do: make output human readable")
    #     # todo: implement
    #     # result of forward method also contains ["classified"]:
    #     # do argmax and convert index back to type?
    #     return

    #@overrides
    #def decode(self):
    #    pass

    # todo: change metric
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # return {precision: 0.9, recall: 0.7, accuracy: 0.8,..}
        # return self._f1.get_metric(reset)
        return self._acc.get_metric(reset)
