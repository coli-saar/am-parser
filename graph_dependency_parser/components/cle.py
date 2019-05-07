from dependency_decoding import chu_liu_edmonds # requires https://github.com/andersjo/dependency_decoding
from allennlp.nn.util import get_device_of

import numpy as np
import torch
from torch.multiprocessing import Pool

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
