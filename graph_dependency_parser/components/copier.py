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
from abc import abstractmethod
from typing import Optional

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.nn import RegularizerApplicator


class Copier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 regularizer: Optional[RegularizerApplicator] = None):
        super().__init__(vocab=vocab, regularizer=regularizer)

    @abstractmethod
    def compute_logprobs(self,
                         encoded_text: torch.Tensor,
                         lemma_copy_tensor: torch.LongTensor,
                         token_copy_tensor: torch.LongTensor,
                         lexlabel_tensor: torch.LongTensor,
                         lexlabel_logits: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def get_most_likely(self, copy_probabilities, lexlabel_id, vocab, lexlabel_namespace, lemma, token):
        raise NotImplementedError()


@Copier.register("lemma_and_token")
class LemmaAndTokenCopier(Copier):

    def __init__(self,
                 mlp: FeedForward,
                 vocab: Vocabulary,
                 regularizer: Optional[RegularizerApplicator] = None):
        super().__init__(vocab=vocab, regularizer=regularizer)
        self.mlp = mlp
        self._encoder_dim = mlp.get_input_dim()
        self.output_layer = torch.nn.Linear(mlp.get_output_dim(), 3)

    def compute_logprobs(self,
                         encoded_text: torch.Tensor,  # dim batch_size x sent_length x mlp_input_dim
                         lemma_copy_tensor: torch.LongTensor,  # dim batch_size x sent_length x sent_length(or later: label list length)
                         token_copy_tensor: torch.LongTensor,  # dim batch_size x sent_length x sent_length(or later: label list length)
                         lexlabel_tensor: torch.LongTensor,  # dim batch_size x sent_length
                         lexlabel_logits: torch.Tensor):  # dim batch_size x sent_length x lexlabel_vocab_size
        print("encoded_text")
        print(encoded_text)
        print("lemma_copy_tensor")
        print(lemma_copy_tensor)
        print("token_copy_tensor")
        print(token_copy_tensor)
        print("lexlabel_tensor")
        print(lexlabel_tensor)
        print("lexlabel_logits")
        print(lexlabel_logits)
        # lexlabel_logits.register_hook(lambda grad: print(f"lexlabel_logits grad: {grad}"))
        output = torch.nn.functional.softmax(self.output_layer(self.mlp(encoded_text)), dim=-1) # dim batch_size x sent_length x 3
        print("output")
        print(output)
        batch_size = lemma_copy_tensor.shape[0]
        sent_length = lemma_copy_tensor.shape[1]
        label_list_length = lemma_copy_tensor.shape[2]
        print("batch_size")
        print(batch_size)
        print("sent_length")
        print(sent_length)
        print("label_list_length")
        print(label_list_length)
        lexlabel_tensor_broadcasted, _ = torch.broadcast_tensors(lexlabel_tensor, torch.zeros(batch_size, sent_length, label_list_length))
        print("lexlabel_tensor_broadcasted")
        print(lexlabel_tensor_broadcasted)
        lexlabel_probs_broadcasted = torch.softmax(lexlabel_logits, dim=-1)
        # lexlabel_probs_broadcasted.register_hook(lambda grad: print(f"lexlabel_probs_broadcasted grad: {grad}"))
        print("lexlabel_probs_broadcasted")
        print(lexlabel_probs_broadcasted)
        vocab_probabilities = torch.gather(lexlabel_probs_broadcasted, 2, lexlabel_tensor_broadcasted)
        # vocab_probabilities.register_hook(lambda grad: print(f"vocab_probabilities grad: {grad}"))
        print("vocab_probabilities")
        print(vocab_probabilities)
        p_vocab = output.narrow(2, 0, 1)
        p_lemma = output.narrow(2, 1, 1)
        p_token = output.narrow(2, 2, 1)
        print("p_vocab")
        print(p_vocab)
        print("p_lemma")
        print(p_lemma)
        print("p_token")
        print(p_token)
        if output.requires_grad:
            output.register_hook(lambda grad: print(f"output grad: {grad}"))
        # p_vocab.register_hook(lambda grad: print(f"p_vocab grad: {grad}"))
        # p_lemma.register_hook(lambda grad: print(f"p_lemma grad: {grad}"))
        # p_token.register_hook(lambda grad: print(f"p_token grad: {grad}"))
        ret = torch.log(p_vocab * vocab_probabilities + p_lemma * lemma_copy_tensor + p_token * token_copy_tensor)
        print("ret")
        print(ret)
        # ret.register_hook(lambda grad: print(f"ret grad: {grad}"))
        return ret, p_vocab, p_lemma, p_token
        # p_vocab, p_lemma, p_token = output
        # vocab_prob = torch.F.softmax(lexlabel_logits)
        # return torch.F.log(p_vocab * vocab_prob + p_lemma * lemma_copy_tensor[token_position][lexlabel_position]
        #                   + p_token * token_copy_tensor[token_position][lexlabel_position])

    def get_most_likely(self,
                        copy_probabilities, # numpy array [p_vocab, p_lemma, p_token]
                        lexlabel_id,
                        vocab,
                        lexlabel_namespace,
                        lemma,
                        token):
        index = copy_probabilities.argmax()
        if index == 0:
            return vocab.get_token_from_index(lexlabel_id, namespace=lexlabel_namespace)
        elif index == 1:
            return lemma
        elif index == 2:
            return token
