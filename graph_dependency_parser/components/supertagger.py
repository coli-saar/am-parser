from typing import Optional

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.nn import RegularizerApplicator


class Supertagger(Model):
    """
    A supertagger, mainly consisting of a MLP.
    """
    def __init__(self,vocab: Vocabulary,
                 mlp : FeedForward,
                 label_namespace : str,
                 regularizer: Optional[RegularizerApplicator] = None ):

        super().__init__(vocab=vocab, regularizer=regularizer)
        self.mlp = mlp
        self._encoder_dim = mlp.get_input_dim()

        self.output_layer = torch.nn.Linear(mlp.get_output_dim(),vocab.get_vocab_size(label_namespace))

    def compute_logits(self, encoded_text : torch.Tensor) -> torch.Tensor:
        """
        Computes class logits for every word in the sentence in the batch.

        :param encoded_text: a tensor of shape (batch_size, seq_len, encoder_dim)
        :return: a tensor of shape (batch_size, seq_len, num_supertag_labels)
        """
        return self.output_layer(self.mlp(encoded_text))

    @staticmethod
    def top_k_supertags(logits: torch.Tensor, k : int) -> torch.Tensor:
        """
        Finds the top k supertags for every word (and every sentence in the batch).
        Does not include scores for supertags.

        :param logits: tensor of shape (batch_size, seq_len, num_supetag_labels)
        :return: tensor of shape (batch_size, seq_len, k)
        """
        assert k > 0, "Number of supertags must be positive"
        #shape (batch_size, seq_len, k)
        top_k = torch.argsort(logits,descending=True,dim=2)[:,:,:k]
        
        return top_k

    def encoder_dim(self):
        return self._encoder_dim




class FragmentSupertagger(Supertagger):
    """
    Convience class to hide the label namespace from the person who write the configuration file.
    """
    def __init__(self,vocab: Vocabulary,
                 mlp : FeedForward,
                 regularizer: Optional[RegularizerApplicator] = None ):
        super().__init__(vocab=vocab,mlp=mlp,regularizer=regularizer,label_namespace="supertag_labels")


class LexlabelTagger(Supertagger):
    """
    Convience class to hide the label namespace from the person who write the configuration file.
    """
    def __init__(self,vocab: Vocabulary,
                 mlp : FeedForward,
                 regularizer: Optional[RegularizerApplicator] = None ):
        super().__init__(vocab=vocab,mlp=mlp,regularizer=regularizer,label_namespace="lex_labels")