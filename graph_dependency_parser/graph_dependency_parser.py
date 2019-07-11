from typing import Dict, Optional, Any, List
import logging

from overrides import overrides
import torch
from torch.nn.modules import Dropout

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from graph_dependency_parser.components.weight_sharer import MTLWeightSharer
from graph_dependency_parser.components.AMTask import AMTask


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
        The (edge) loss function to be used.
    supertagger: ``components.supertagger.FragmentSupertagger``, required.
        The supertagging model that predicts graph constants (graph fragments + types)
    lexlabeltagger: ``components.supertagger.LexlabelTagger``, required.
        The supertagging model that predicts lexical labels for the supertags.
    supertagger_loss: ``components.losses.supertagging.SupertaggingLoss``, required.
        The loss function for the supertagging model.
    lexlabel_loss: ``components.losses.supertagging.SupertaggingLoss``, required.
        The loss function for the lexical label tagger.
    loss_mixing : Dict[str,float] = None,
        The mixing coefficients for the different losses. Valid loss names are "edge_existence",
        "edge_label","supertagging" and "lexlabel".

    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    lemma_embedding : ``Embedding``, optional.
        Used to embed the ``lemmas`` ``SequenceLabelField`` we get as input to the model.
    ne_embedding : ``Embedding``, optional.
        Used to embed the ``ner_labels`` ``SequenceLabelField`` we get as input to the model.
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
    validation_evaluator: ``ValidationEvaluator``, optional (default=``None``)
        If provided, will be used to compute external validation metrics after each epoch.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: MTLWeightSharer,
                 tasks: List[AMTask],
                 pos_tag_embedding: Embedding = None,
                 lemma_embedding: Embedding = None,
                 ne_embedding: Embedding = None,
                 input_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(GraphDependencyParser, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder


        self._pos_tag_embedding = pos_tag_embedding or None
        self._lemma_embedding = lemma_embedding
        self._ne_embedding = ne_embedding

        self._input_dropout = Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder.get_output_dim()]))

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()
        if self._lemma_embedding is not None:
            representation_dim += lemma_embedding.get_output_dim()
        if self._ne_embedding is not None:
            representation_dim += ne_embedding.get_output_dim()

        assert len(tasks) > 0, "List of tasks must not be empty"
        self.tasks : Dict[str, AMTask] = {t.name : t for t in tasks}

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        for t in tasks:
            t.check_all_dimensions_match(encoder.get_output_dim())

        for formalism,task in sorted(self.tasks.items(), key=lambda nt: nt[0]):
            #sort by name of formalism for consistent ordering
            self.add_module(formalism,task)
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
        if 'formalism' not in metadata[0]:
            raise ConfigurationError("metadata is missing 'formalism' key.\
            Please use the amconll dataset reader.")

        formalism_of_batch = metadata[0]['formalism']
        for entry in metadata:
            if entry['formalism'] != formalism_of_batch:
                raise ConfigurationError("Two formalisms in the same batch.")
        if not formalism_of_batch in self.tasks.keys():
            raise ConfigurationError(f"Got formalism {formalism_of_batch} but I only have these tasks: {list(self.tasks.keys())}")

        embedded_text_input = self.text_field_embedder(words)
        concatenated_input = [embedded_text_input]
        if pos_tags is not None and self._pos_tag_embedding is not None:
            concatenated_input.append(self._pos_tag_embedding(pos_tags))
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        if self._lemma_embedding is not None:
            concatenated_input.append(self._lemma_embedding(lemmas))
        if self._ne_embedding is not None:
            concatenated_input.append(self._ne_embedding(ner_tags))

        if len(concatenated_input) > 1:
            embedded_text_input = torch.cat(concatenated_input, -1)
        mask = get_text_field_mask(words)
        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text_parsing, encoded_text_tagging = self.encoder(formalism_of_batch, embedded_text_input, mask) #potentially weight-sharing

        batch_size, seq_len, encoding_dim = encoded_text_parsing.size()
        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the artificial root onto the sentence representation.
        encoded_text_parsing = torch.cat([head_sentinel, encoded_text_parsing], 1)

        if encoded_text_tagging is not None: #might be none when batch is of formalism without tagging (UD)
            batch_size, seq_len, encoding_dim = encoded_text_tagging.size()
            head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
            # Concatenate the artificial root onto the sentence representation.
            encoded_text_tagging = torch.cat([head_sentinel, encoded_text_tagging], 1)

        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)

        return self.tasks[formalism_of_batch](encoded_text_parsing, encoded_text_tagging, mask, pos_tags, metadata, supertags, lexlabels, head_tags, head_indices)


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        """
        In contrast to its name, this function does not perform the decoding but only prepares it.
        Therefore, we take the result of forward and perform the following steps (for each sentence in batch):
        - remove padding
        - identifiy the root of the sentence, group other root-candidates under the proper root
        - collect a selection of supertags to speed up computation (top k selection is done later)
        :param output_dict: result of forward
        :return: output_dict with the following keys added:
            - lexlabels: nested list: contains for each sentence, for each word the most likely lexical label (w/o artificial root)
            - supertags: nested list: contains for each sentence, for each word the most likely lexical label (w/o artificial root)
        """
        formalism = output_dict.pop("formalism")

        return self.tasks[formalism].decode(output_dict)


    @overrides
    def get_metrics(self, reset: bool = False, model_path = None) -> Dict[str, float]:
        r = dict()
        for name,task in self.tasks.items():
            for metric, val in task.metrics(parser_model=self, reset=reset, model_path=model_path).items():
                r[name+"_"+metric] = val
        return r
