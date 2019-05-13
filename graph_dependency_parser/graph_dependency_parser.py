from typing import Dict, Optional, Any, List
import logging

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import AttachmentScores

import graph_dependency_parser.components.losses
import graph_dependency_parser.components.edge_models

from graph_dependency_parser.components.cle import  cle_decode
from graph_dependency_parser.components.evaluation.predictors import  ValidationEvaluator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

POS_TO_IGNORE = {'``', "''", ':', ',', '.', 'PU', 'PUNCT', 'SYM'}

@Model.register("graph_dependency_parser")
class GraphDependencyParser(Model):
    """
    This dependency graph_dependency_parser is a blueprint for several graph-based dependency parsers.

    There are several possible edge models and loss functions.

    For decoding, the CLE algorithm is used (during training attachments scores are usually based on greedy decoding)

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    edge_model: ``components.edge_models.EdgeModel``, required.
        The edge model to be used.
    loss_function: ``components.losses.EdgeLoss``, required.
        The loss function to be used.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    use_mst_decoding_for_validation : ``bool``, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
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
                 edge_model: graph_dependency_parser.components.edge_models.EdgeModel,
                 loss_function: graph_dependency_parser.components.losses.EdgeLoss,
                 pos_tag_embedding: Embedding = None,
                 use_mst_decoding_for_validation: bool = True,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 validation_evaluator: Optional[ValidationEvaluator] = None) -> None:
        super(GraphDependencyParser, self).__init__(vocab, regularizer)

        self.validation_evaluator = validation_evaluator

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        self._pos_tag_embedding = pos_tag_embedding or None
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder.get_output_dim()]))

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(encoder.get_output_dim(), edge_model.encoder_dim(),
                               "encoder output dim", "input dim edge model")

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        tags = self.vocab.get_token_to_index_vocabulary("pos")
        punctuation_tag_indices = {tag: index for tag, index in tags.items() if tag in POS_TO_IGNORE}
        self._pos_to_ignore = set(punctuation_tag_indices.values())
        logger.info(f"Found POS tags corresponding to the following punctuation : {punctuation_tag_indices}. "
                    "Ignoring words with these POS tags for evaluation.")

        self._attachment_scores = AttachmentScores()
        initializer(self)

        self.edge_model = edge_model
        self.loss_function = loss_function

        #Being able to detect what state we are in, probably not the best idea.
        self.current_epoch = 1
        self.pass_over_data_just_started = True

    @overrides
    def forward(self,  # type: ignore
                words: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor,
                metadata: List[Dict[str, Any]],
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
            A torch tensor representing the sequence of integer gold edge labels for the arcs
            in the dependency parse. Has shape ``(batch_size, sequence_length)``.
        head_indices : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length)``.

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        arc_loss : ``torch.FloatTensor``
            The loss contribution from the unlabeled arcs.
        edge_label_loss : ``torch.FloatTensor``
            The loss contribution from the edge labels.
        heads : ``torch.FloatTensor``
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        edge_labels : ``torch.FloatTensor``
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.
        """
        embedded_text_input = self.text_field_embedder(words)
        if pos_tags is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(words)
        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)
        encoded_text = self._dropout(encoded_text)

        edge_existence_scores = self.edge_model.edge_existence(encoded_text,mask)

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads = self._greedy_decode_arcs(edge_existence_scores,mask)
            edge_label_logits = self.edge_model.label_scores(encoded_text, predicted_heads)
            predicted_edge_labels = self._greedy_decode_edge_labels(edge_label_logits)
        else:
            #Find best tree with CLE
            predicted_heads = cle_decode(edge_existence_scores,mask.data.sum(dim=1).long())
            #With info about tree structure, get edge label scores
            edge_label_logits = self.edge_model.label_scores(encoded_text, predicted_heads)
            #Predict edge labels
            predicted_edge_labels = self._greedy_decode_edge_labels(edge_label_logits)

        output_dict = {
            "heads": predicted_heads,
            "edge_labels": predicted_edge_labels,
            "mask": mask,
            "words": [meta["words"] for meta in metadata],
            "pos": [meta["pos"] for meta in metadata],
            "position_in_corpus": [meta["position_in_corpus"] for meta in metadata],
        }

        if head_indices is not None and head_tags is not None:
            gold_edge_label_logits = self.edge_model.label_scores(encoded_text, head_indices)
            edge_label_loss = self.loss_function.label_loss(gold_edge_label_logits, mask, head_tags)

            arc_nll = self.loss_function.edge_existence_loss(edge_existence_scores,head_indices,mask)

            loss = arc_nll + edge_label_loss

            evaluation_mask = self._get_mask_for_eval(mask[:, 1:], pos_tags)
            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachment_scores(predicted_heads[:, 1:],
                                    predicted_edge_labels[:, 1:],
                                    head_indices[:, 1:],
                                    head_tags[:, 1:],
                                    evaluation_mask)
            output_dict["arc_loss"] = arc_nll
            output_dict["edge_label_loss"] = edge_label_loss
            output_dict["loss"] = loss

        if self.pass_over_data_just_started:
            # here we could decide if we want to start collecting info for the decoder.
            pass
        self.pass_over_data_just_started = False
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        """
        Takes the result of forward and creates human readable, non-padded dependency trees.
        :param output_dict:
        :return: output_dict with two new keys, "predicted_labels" and "predicted_heads", which are lists of lists.
        """

        head_tags = output_dict.pop("edge_labels").cpu().detach().numpy()
        heads = output_dict.pop("heads").cpu().detach().numpy()
        mask = output_dict.pop("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)
        edge_labels = []
        head_indices = []
        for instance_heads, instance_tags, length in zip(heads, head_tags, lengths):
            instance_heads = list(instance_heads[1:length])
            instance_tags = instance_tags[1:length]
            labels = [self.vocab.get_token_from_index(label, "head_tags")
                      for label in instance_tags]
            edge_labels.append(labels)
            head_indices.append(instance_heads)

        output_dict["predicted_labels"] = edge_labels
        output_dict["predicted_heads"] = head_indices
        return output_dict


    def _greedy_decode_edge_labels(self, edge_label_logits: torch.Tensor) -> torch.Tensor:
        """
        Assigns edge labels according to (existing) edges.
        Parameters
        ----------
        edge_label_logits: ``torch.Tensor`` of shape (batch_size, sequence_length, num_head_tags)

        Returns
        -------
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded head tags (labels of incoming edges) of each word.
        """
        _, head_tags = edge_label_logits.max(dim=2)
        return head_tags

    def _greedy_decode_arcs(self,
                            existence_scores: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """
        Decodes the head  predictions by decoding the unlabeled arcs
        independently for each word. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        existence_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        mask: torch.Tensor, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        existence_scores = existence_scores + torch.diag(existence_scores.new(mask.size(1)).fill_(-numpy.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).byte().unsqueeze(2)
            existence_scores.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = existence_scores.max(dim=2)
        return heads


    def _get_mask_for_eval(self,
                           mask: torch.LongTensor,
                           pos_tags: torch.LongTensor) -> torch.LongTensor:
        """
        Dependency evaluation excludes words are punctuation.
        Here, we create a new mask to exclude word indices which
        have a "punctuation-like" part of speech tag.

        Parameters
        ----------
        mask : ``torch.LongTensor``, required.
            The original mask.
        pos_tags : ``torch.LongTensor``, required.
            The pos tags for the sequence.

        Returns
        -------
        A new mask, where any indices equal to labels
        we should be ignoring are masked.
        """
        new_mask = mask.detach()
        for label in self._pos_to_ignore:
            label_mask = pos_tags.eq(label).long()
            new_mask = new_mask * (1 - label_mask)
        return new_mask

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        r = self._attachment_scores.get_metric(reset)
        if reset:
            if self.training: #done on the training data
                pass
                self.current_epoch += 1
                self.pass_over_data_just_started = True
            else: #done on dev/test data
                if self.validation_evaluator:
                    metrics = self.validation_evaluator.eval(self)
                    for name, val in metrics.items():
                        r[name] = val
        return r
