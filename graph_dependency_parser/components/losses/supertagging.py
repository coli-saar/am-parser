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