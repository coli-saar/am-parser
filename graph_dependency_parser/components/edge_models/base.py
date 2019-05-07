
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
