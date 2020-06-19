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

import torch

from allennlp.models.model import Model

from abc import abstractmethod



class EdgeModel(Model):
    """
    Interface for edge models.
    """
    @abstractmethod
    def edge_existence(self, encoded_text: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes edge existence scores for a batch of sentences
        :param encoded_text: torch.Tensor of shape (batch_size, sequence length, encoding dim). Represents the input sentence, with artifical root node ('head sentinel') added to the beginning of the sequence.
        :param mask: mask : torch.Tensor of shape (batch_size, sequence_length), denoting unpadded elements in the sequence.
        :return: a tensor of shape (batch_size, sequence length, sequence length)
        """
        raise NotImplementedError()

    @abstractmethod
    def label_scores(self, encoded_text: torch.Tensor, head_indices: torch.Tensor) -> torch.Tensor:
        """
        Returns edge label scores given the encoded text and the tree structure
        :param encoded_text: torch.Tensor of shape (batch_size, sequence length, encoding dim). Represents the input sentence, with artifical root node ('head sentinel') added to the beginning of the sequence.
        :param head_indices:  tensor of shape (batch_size, sequence_length). The indices of the heads for every word (predicted or gold), i.e. represents dependency trees.
        :return: A tensor of shape (batch_size, sequence_length, num_head_tags), representing logits for predicting a distribution over tags for each edge.
        """
        raise NotImplementedError()


    @abstractmethod
    def encoder_dim(self):
        """
        :return: the output dim of the encoder used, must match encoding dim in edge_existence
        """
        raise NotImplementedError()

    @abstractmethod
    def full_label_scores(self, encoded_text:torch.Tensor) -> torch.Tensor:
        """
        Computes edge label scores for all edges for a batch of sentences.

        Parameters
        ----------
         encoded_text : torch.Tensor, required
            The input sentence, with artifical root node (head sentinel) added in the beginning of
            shape (batch_size, sequence length, encoding dim)

        Returns
        -------
        edge_label_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length,sequence_length, num_edge_labels),
            representing logits for predicting a distribution over edge labels
            for each edge. [i,j,k,l] is the the score for edge j->k being labeled l in sentence i
        """
        raise NotImplementedError()
