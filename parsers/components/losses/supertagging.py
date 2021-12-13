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
from allennlp.common import Registrable
from allennlp.nn.util import sequence_cross_entropy_with_logits


class SupertaggingLoss(Registrable):
    """
    Softmax cross entropy loss, made usable for configuration files.
    """
    def __init__(self, normalize_wrt_seq_len : bool = False, label_smoothing : int = None):
        super().__init__()
        self.normalize_wrt_seq_len = normalize_wrt_seq_len
        self.label_smoothing = label_smoothing

    def loss(self,logits: torch.Tensor, gold_labels : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        """
        Computes the loss.
        :param logits: tensor of shape (batch_size, seq_len, num_classes)
        :param gold_labels: tensor of shape (batch_size, seq_len)
        :param mask: tensor of shape (batch_size, seq_len)
        :return:
        """
        l = sequence_cross_entropy_with_logits(logits,gold_labels,mask, label_smoothing=self.label_smoothing)

        #sequence_cross entropy automatically normalizes by batch, so multiply by batch again
        if not self.normalize_wrt_seq_len:
            l *= logits.size(0)
        return l