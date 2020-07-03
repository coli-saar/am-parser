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
from typing import List, Tuple

from dependency_decoding import chu_liu_edmonds # requires https://github.com/andersjo/dependency_decoding
from allennlp.nn.util import get_device_of

import numpy as np
import torch

def cle_decode(scores, lengths):
    """
    Parses a batch of sentences
    :param scores torch.Tensor of shape (batch_size,tokens, tokens), the length of the sentences is an array of length batch_size that specifies how long the sentences are
    :param lengths: actual lengths of the sentences, tensor of shape (batch_size,)
    :return: a tensor of shape (batch_size, tokens) that contains the heads of the tokens. Positions that go over the sentence length are filled with -1.
    """
    heads = []
    scores = scores.detach().cpu().double().numpy()
    lengths = lengths.cpu().numpy()
    bs, toks, _ = scores.shape
    for m,l in zip(scores,lengths):
        r,_ = chu_liu_edmonds(m[:l,:l]) #discard _score_ of solution
        h = np.concatenate([r, -np.ones(toks-l,dtype=np.long)])
        heads.append(h)
    return torch.from_numpy(np.stack(heads))

def get_head_dict(heads):
    """
    Takes a list of heads for a sentence and returns a dictionary that maps words to the set with their children
    :param heads:
    :return:
    """
    #usually, you want to call get_head_dict(some_heads[1:]) #strip off -1
    head_dict = dict()
    for (m,h) in enumerate(heads):
        if h not in head_dict:
            head_dict[h] = set()
        head_dict[h].add(m+1)
    return head_dict


def cle_loss(scores: torch.Tensor, lengths : torch.Tensor, gold_heads : torch.Tensor, normalize_wrt_seq_len : bool):
    """
        Parses a batch of sentences and computes a hinge loss (see code by Eliyahu Kiperwasser: https://github.com/elikip/bist-parser)
        :param scores torch.Tensor of shape (batch_size,tokens, tokens), the length of the sentences is an array of length batch_size that specifies how long the sentences are
        :param gold_heads: Tensor of shape (batch_size, tokens) that contains the correct head for every word.
        :param lengths: actual lengths of the sentences, tensor of shape (batch_size,)
        :return: a scalar torch.Tensor with the hinge loss
        """
    losses : torch.Tensor = 0
    device = get_device_of(scores)
    scores = scores.cpu()
    #scores_np = scores.detach().double().numpy()

    gold_heads = gold_heads.cpu().numpy()
    lengths = lengths.cpu().numpy()

    for m,g,l in zip(scores,gold_heads,lengths):
        #m: shape (tokens, tokens)
        #g: shape (tokens,)
        #l: scalar, sentence length
        range = np.arange(l)
        #remove padding at the end:
        m = m[:l, :l]
        g = g[:l]  # -> shape (l,)

        # make gold solution look worse by cost augmentation (in the original, make non-gold look better)/introduce margin:
        m[range, g] -= 1.0 # cost augmentation

        r,_ = chu_liu_edmonds(m.detach().double().numpy()) #discard _score_ of solution, -> r has shape (l,)
        # this implementation says that head of artificial root is -1, but the rest of the pipeline says the head of the artificial root is the artificial root itself (i.e. 0):
        r[0] = 0
        r = np.array(r)

        scores_of_solution = m[range,r] #extract the scores belonging to the decoded edges -> shape (l,)
        scores_of_gold = m[range,g] # extract the scores belonging to the gold edges -> shape (l,)
        r = torch.from_numpy(r)
        g = torch.from_numpy(g)
        zero = torch.zeros(1,dtype=torch.float32)
        #where predicted head differs from gold head, add the score difference to the loss term:
        loss_term = torch.sum(torch.where(torch.eq(r,g), zero, scores_of_solution-scores_of_gold))
        if normalize_wrt_seq_len:
            loss_term /= l
        losses += loss_term
    if device < 0:
        return losses
    return losses.to(device)

def find_root(heads : List[int], best_supertags : List[int], label_scores:np.array, root_edge_label_id : int, bot_id : int, modify : bool) -> Tuple[List[int],int]:
    """
    Selects the root and potentially changes some attachments. We take everything attached to the artificial root (index 0) and regroup it under the actual root.
    Exempted from this are words whose highest scoring supertag is \bot.
    We find the root by looking at those children of index 0 that have the highest edge label scoring for being ROOT.
    :param heads: a list of sentence length that gives the head of each position
    :param best_supertags: a list of sentence length with highest scoring supertags (as ints)
    :param label_scores: a numpy array of shape (sentence length, num of labels)
        that contains the scores for the edge labels on the edges given in heads.
    :param root_edge_label_id: the id of the edge label ROOT from the vocabulary
    :param bot_id: the id of the supertag \bot from the vocabulary.
    :param modify: change the heads? Or only find the root?

    :return: return the (modified) list of heads and the index of the actual root.
    """
    assert len(best_supertags) == len(heads)
    assert label_scores.shape[0] == len(heads)
    head_dict = get_head_dict(heads)
    #those positions that are attached to 0 and whose best supertag is not \bot. If the supertag is bot, then we only look at those that are heads themselves
    attached_to_0: List[int] = [index for index in head_dict[0] if best_supertags[index-1] != bot_id or index in head_dict]
    if len(attached_to_0) > 0:
        root_scores = []
        for dependent_of_0 in attached_to_0:
            root_scores.append(label_scores[dependent_of_0-1,root_edge_label_id])
        new_root_id : int = attached_to_0[np.argmax(np.array(root_scores))]
        if modify:
            for e in attached_to_0:
                if e != new_root_id:
                    heads[e-1] = new_root_id
    else:
        if len(heads) == 1:  # single word sentence
            new_root_id = 1 #1-based
        else:
            attached_to_0 = list(head_dict[0])
            print("WARNING: choosing root node arbitrarily!")
            if attached_to_0:
                new_root_id = attached_to_0[0] #any element
            else:
                raise ValueError("Nothing attached to 0?")

    return heads, new_root_id