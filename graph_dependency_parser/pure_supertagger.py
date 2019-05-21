from typing import Dict, Optional, Any, List
import logging

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import torch.nn.functional as F
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout, FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, get_device_of, masked_log_softmax
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import AttachmentScores, SequenceAccuracy, CategoricalAccuracy

import graph_dependency_parser.components.losses
import graph_dependency_parser.components.edge_models

from graph_dependency_parser.components.cle import cle_decode, find_root
from graph_dependency_parser.components.evaluation.predictors import ValidationEvaluator, AMconllPredictor
from graph_dependency_parser.components.losses.supertagging import SupertaggingLoss
from graph_dependency_parser.components.supertagger import FragmentSupertagger, LexlabelTagger, Supertagger

from graph_dependency_parser.components.dataset_readers.amconll_tools import AMSentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("pure_supertagger")
class SimpleSupertagger(Model):
    """
    This module is a supertagger and is intended to make search for good hyperparameters for the integrated supertagger faster.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    supertagger: ``components.supertagger.FragmentSupertagger``, required.
        The supertagging model that predicts graph constants (graph fragments + types)
    lexlabeltagger: ``components.supertagger.LexlabelTagger``, required.
        The supertagging model that predicts lexical labels for the supertags.
    supertagger_loss: ``components.losses.supertagging.SupertaggingLoss``, required.
        The loss function for the supertagging model.
    lexlabel_loss: ``components.losses.supertagging.SupertaggingLoss``, required.
        The loss function for the lexical label tagger.
    loss_mixing : Dict[str,float] = None,
        The mixing coefficients for the different losses. Valid loss names are "supertagging" and "lexlabel".

    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    lemma_embedding : ``Embedding``, optional.
        Used to embed the ``lemmas`` ``SequenceLabelField`` we get as input to the model.
    ne_embedding : ``Embedding``, optional.
        Used to embed the ``ner_labels`` ``SequenceLabelField`` we get as input to the model.
    dropout : ``float``, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : ``float``, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 supertagger: FragmentSupertagger,
                 lexlabeltagger: LexlabelTagger,
                 supertagger_loss : SupertaggingLoss,
                 lexlabel_loss : SupertaggingLoss,
                 loss_mixing : Dict[str,float] = None,
                 pos_tag_embedding: Embedding = None,
                 lemma_embedding: Embedding = None,
                 ne_embedding: Embedding = None,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SimpleSupertagger, self).__init__(vocab, regularizer)


        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.loss_mixing = loss_mixing or dict()

        self._pos_tag_embedding = pos_tag_embedding or None
        self._lemma_embedding = lemma_embedding
        self._ne_embedding = ne_embedding
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()
        if self._lemma_embedding is not None:
            representation_dim += lemma_embedding.get_output_dim()
        if self._ne_embedding is not None:
            representation_dim += ne_embedding.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(encoder.get_output_dim(), supertagger.encoder_dim(),
                               "encoder output dim", "supertagger input dim")
        check_dimensions_match(encoder.get_output_dim(), lexlabeltagger.encoder_dim(),
                               "encoder output dim", "lexical label tagger input dim")
        loss_names = ["supertagging","lexlabel"]
        for loss_name in loss_names:
            if loss_name not in self.loss_mixing:
                self.loss_mixing[loss_name] = 1.0
                logger.info(f"Loss name {loss_name} not found in loss_mixing, using a weight of 1.0")
        not_contained = set(self.loss_mixing.keys()) - set(loss_names)
        if len(not_contained):
            logger.critical(f"The following loss name(s) are unknown: {not_contained}")

        self.supertagger = supertagger
        self.lexlabeltagger = lexlabeltagger


        self.supertagger_loss = supertagger_loss
        self.lexlabel_loss = lexlabel_loss

        self._supertagging_acc = CategoricalAccuracy()
        self._supertagging_acc3 = CategoricalAccuracy(top_k=3)
        self._lexlabel_acc = CategoricalAccuracy()
        self._lexlabel_acc3 = CategoricalAccuracy(top_k=3)

        initializer(self)


    @overrides
    def forward(self,  # type: ignore
                words: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor,
                lemmas: torch.LongTensor,
                ner_tags: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                supertags: torch.LongTensor = None,
                lexlabels: torch.LongTensor = None,
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        words : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, sequence_length)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        pos_tags : ``torch.LongTensor``, required
            The output of a ``SequenceLabelField`` containing POS tags.
            POS tags are required regardless of whether they are used in the model,
            because they are used to filter the evaluation metric to only consider
            heads of words which are not punctuation.
        metadata : List[Dict[str, Any]], optional (default=None)
            A dictionary of metadata for each batch element which has keys:
                words : ``List[str]``, required.
                    The tokens in the original sentence.
                pos : ``List[str]``, required.
                    The dependencies POS tags for each word.
        head_tags : = edge_labels torch.LongTensor, optional (default = None)
            Ignored
        head_indices : torch.LongTensor, optional (default = None)
            Ignored

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.
        """
        embedded_text_input = self.text_field_embedder(words)
        concatenated_input = [embedded_text_input]
        if pos_tags is not None:
            concatenated_input.append(self._pos_tag_embedding(pos_tags))
        if self._lemma_embedding is not None:
            concatenated_input.append(self._lemma_embedding(lemmas))
        if self._ne_embedding is not None:
            concatenated_input.append(self._ne_embedding(ner_tags))
        if len(concatenated_input) > 1:
            embedded_text_input = torch.cat(concatenated_input, -1)

        mask = get_text_field_mask(words)
        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        encoded_text = self._dropout(encoded_text)

        supertagger_logits = self.supertagger.compute_logits(encoded_text)  # shape (batch_size, seq_len, num_supertags)
        lexlabel_logits = self.lexlabeltagger.compute_logits(encoded_text)  # shape (batch_size, seq_len, num label tags)

        output_dict = {
            "supertag_scores" : supertagger_logits, # shape (batch_size, seq_len, num supertags)
            "lexlabels" : Supertagger.top_k_supertags(lexlabel_logits,1).squeeze(2), #shape (batch_size, seq_len)
            "mask": mask,
            "words": [meta["words"] for meta in metadata],
            "attributes": [ meta["attributes"] for meta in metadata],
            "position_in_corpus": [meta["position_in_corpus"] for meta in metadata],
        }

        #Compute loss:
        if supertags is not None and lexlabels is not None:
            supertagging_nll = self.supertagger_loss.loss(supertagger_logits, supertags, mask)
            lexlabel_nll = self.lexlabel_loss.loss(lexlabel_logits, lexlabels, mask)

            loss = self.loss_mixing["supertagging"]*supertagging_nll \
                   + self.loss_mixing["lexlabel"] * lexlabel_nll
            # compare against gold data
            self._supertagging_acc(supertagger_logits, supertags, mask)
            self._lexlabel_acc(lexlabel_logits, lexlabels, mask)
            self._supertagging_acc3(supertagger_logits,supertags,mask)
            self._lexlabel_acc3(lexlabel_logits,lexlabels,mask)

            output_dict["supertagging_loss"] = supertagging_nll
            output_dict["lexlabel_loss"] = lexlabel_nll
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        r = dict()
        r["Supertagging Acc"] = self._supertagging_acc.get_metric(reset)
        r["Supertagging Acc (top 3)"] = self._supertagging_acc3.get_metric(reset)
        r["Lex Label Acc"] = self._lexlabel_acc.get_metric(reset)
        r["Lex Label Acc (top 3)"] = self._lexlabel_acc3.get_metric(reset)
        return r
