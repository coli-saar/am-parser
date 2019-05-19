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
                 encoder: Seq2SeqEncoder,
                 edge_model: graph_dependency_parser.components.edge_models.EdgeModel,
                 loss_function: graph_dependency_parser.components.losses.EdgeLoss,
                 supertagger: FragmentSupertagger,
                 lexlabeltagger: LexlabelTagger,
                 supertagger_loss : SupertaggingLoss,
                 lexlabel_loss : SupertaggingLoss,
                 loss_mixing : Dict[str,float] = None,
                 pos_tag_embedding: Embedding = None,
                 lemma_embedding: Embedding = None,
                 ne_embedding: Embedding = None,
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
        self.loss_mixing = loss_mixing or dict()

        self._pos_tag_embedding = pos_tag_embedding or None
        self._lemma_embedding = lemma_embedding
        self._ne_embedding = ne_embedding
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder.get_output_dim()]))

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()
        if self._lemma_embedding is not None:
            representation_dim += lemma_embedding.get_output_dim()
        if self._ne_embedding is not None:
            representation_dim += ne_embedding.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(encoder.get_output_dim(), edge_model.encoder_dim(),
                               "encoder output dim", "input dim edge model")
        check_dimensions_match(encoder.get_output_dim(), supertagger.encoder_dim(),
                               "encoder output dim", "supertagger input dim")
        check_dimensions_match(encoder.get_output_dim(), lexlabeltagger.encoder_dim(),
                               "encoder output dim", "lexical label tagger input dim")
        loss_names = ["edge_existence","edge_label","supertagging","lexlabel"]
        for loss_name in loss_names:
            if loss_name not in self.loss_mixing:
                self.loss_mixing[loss_name] = 1.0
                logger.info(f"Loss name {loss_name} not found in loss_mixing, using a weight of 1.0")
        not_contained = set(self.loss_mixing.keys()) - set(loss_names)
        if len(not_contained):
            logger.critical(f"The following loss name(s) are unknown: {not_contained}")

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        tags = self.vocab.get_token_to_index_vocabulary("pos")
        punctuation_tag_indices = {tag: index for tag, index in tags.items() if tag in POS_TO_IGNORE}
        self._pos_to_ignore = set(punctuation_tag_indices.values())
        logger.info(f"Found POS tags corresponding to the following punctuation : {punctuation_tag_indices}. "
                    "Ignoring words with these POS tags for evaluation.")

        self.supertagger = supertagger
        self.lexlabeltagger = lexlabeltagger

        self.edge_model = edge_model
        self.loss_function = loss_function

        self.supertagger_loss = supertagger_loss
        self.lexlabel_loss = lexlabel_loss

        self._supertagging_acc = CategoricalAccuracy()
        self._lexlabel_acc = CategoricalAccuracy()
        self._attachment_scores = AttachmentScores()
        initializer(self)

        self.current_epoch = 0


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
        encoded_text = self.encoder(embedded_text_input, mask)

        batch_size, seq_len, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the artificial root onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)

        encoded_text = self._dropout(encoded_text)

        #TODO: keep in mind that softmax is not applied before decoding!
        edge_existence_scores = self.edge_model.edge_existence(encoded_text,mask) #shape (batch_size, seq_len, seq_len)
        supertagger_logits = self.supertagger.compute_logits(encoded_text)  # shape (batch_size, seq_len, num_supertags)
        lexlabel_logits = self.lexlabeltagger.compute_logits(encoded_text)  # shape (batch_size, seq_len, num label tags)

        #Make predictions on data:
        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads = self._greedy_decode_arcs(edge_existence_scores,mask)
            edge_label_logits = self.edge_model.label_scores(encoded_text, predicted_heads) # shape (batch_size, seq_len, num edge labels)
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
            "label_logits" : edge_label_logits, # shape (batch_size, seq_len, num edge labels)
            "supertag_scores" : supertagger_logits, # shape (batch_size, seq_len, num supertags)
            "best_supertags" : Supertagger.top_k_supertags(supertagger_logits,1).squeeze(2), #shape (batch_size, seq_len)
            "lexlabels" : Supertagger.top_k_supertags(lexlabel_logits,1).squeeze(2), #shape (batch_size, seq_len)
            "mask": mask,
            "words": [meta["words"] for meta in metadata],
            "attributes": [ meta["attributes"] for meta in metadata],
            "encoded_text" : encoded_text,
            "position_in_corpus": [meta["position_in_corpus"] for meta in metadata],
        }

        #Compute loss:
        if head_indices is not None and head_tags is not None:
            gold_edge_label_logits = self.edge_model.label_scores(encoded_text, head_indices)
            edge_label_loss = self.loss_function.label_loss(gold_edge_label_logits, mask, head_tags)

            edge_existence_loss = self.loss_function.edge_existence_loss(edge_existence_scores,head_indices,mask)
            #compute loss, remove loss for artificial root
            supertagger_logits = supertagger_logits[:,1:,:].contiguous()
            lexlabel_logits = lexlabel_logits[:,1:,:].contiguous()

            supertagging_nll = self.supertagger_loss.loss(supertagger_logits, supertags, mask[:,1:])
            lexlabel_nll = self.lexlabel_loss.loss(lexlabel_logits, lexlabels, mask[:,1:])

            loss = self.loss_mixing["edge_existence"]*edge_existence_loss \
                   + self.loss_mixing["edge_label"]*edge_label_loss \
                   + self.loss_mixing["supertagging"]*supertagging_nll \
                   + self.loss_mixing["lexlabel"]* lexlabel_nll

            #Compute LAS/UAS/Supertagging acc/Lex label acc:
            evaluation_mask = self._get_mask_for_eval(mask[:, 1:], pos_tags)
            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachment_scores(predicted_heads[:, 1:],
                                    predicted_edge_labels[:, 1:],
                                    head_indices[:, 1:],
                                    head_tags[:, 1:],
                                    evaluation_mask)
            self._supertagging_acc(supertagger_logits, supertags, mask[:,1:]) #compare against gold data
            self._lexlabel_acc(lexlabel_logits, lexlabels, mask[:,1:]) #compare against gold data

            output_dict["arc_loss"] = edge_existence_loss
            output_dict["edge_label_loss"] = edge_label_loss
            output_dict["supertagging_loss"] = supertagging_nll
            output_dict["lexlabel_loss"] = lexlabel_nll
            output_dict["loss"] = loss
        return output_dict

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

        best_supertags = output_dict.pop("best_supertags").cpu().detach().numpy()
        supertag_scores = output_dict.pop("supertag_scores") # shape (batch_size, seq_len, num supertags)
        k = 10
        if self.validation_evaluator: #retrieve k supertags from validation evaluator.
            if isinstance(self.validation_evaluator.predictor,AMconllPredictor):
                k = self.validation_evaluator.predictor.k
        top_k_supertags = Supertagger.top_k_supertags(supertag_scores, k).cpu().detach().numpy() # shape (batch_size, seq_len, k)
        supertag_scores = supertag_scores.cpu().detach().numpy()
        lexlabels = output_dict.pop("lexlabels").cpu().detach().numpy() #shape (batch_size, seq_len)
        heads = output_dict.pop("heads")
        heads_cpu = heads.cpu().detach().numpy()
        mask = output_dict.pop("mask")
        edge_label_logits = output_dict.pop("label_logits").cpu().detach().numpy()  # shape (batch_size, seq_len, num edge labels)
        encoded_text = output_dict.pop("encoded_text")
        lengths = get_lengths_from_binary_sequence_mask(mask)

        #here we collect things:
        all_edge_label_logits = []
        all_supertags = []
        head_indices = []
        roots = []
        all_predicted_lex_labels = []

        #we need the following to identify the root
        root_edge_label_id = self.vocab.get_token_index("ROOT",namespace="head_tags")
        bot_id = self.vocab.get_token_index(AMSentence.get_bottom_supertag(),namespace="supertag_labels")

        for i, length in enumerate(lengths):
            instance_heads_cpu = list(heads_cpu[i,1:length])
            #Postprocess heads and find root of sentence:
            instance_heads_cpu, root = find_root(instance_heads_cpu, best_supertags[i,1:length], edge_label_logits[i,1:length,:], root_edge_label_id, bot_id, modify=True)
            roots.append(root)
            #apply changes to instance_heads tensor:
            instance_heads = heads[i,:]
            for j, x in enumerate(instance_heads_cpu):
                instance_heads[j+1] = torch.tensor(x) #+1 because we removed the first position from instance_heads_cpu

            # re-calculate edge label logits since heads might have changed:
            label_logits = self.edge_model.label_scores(encoded_text[i].unsqueeze(0), instance_heads.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            #(un)squeeze: fake batch dimension
            all_edge_label_logits.append(label_logits[1:length,:])

            #calculate supertags for this sentence:
            supertags_for_this_sentence = []
            for word in range(1,length): #TODO allow only ART-ROOT supertags at ART-ROOT
                supertags_for_this_word = []
                for top_k in top_k_supertags[i,word]:
                    fragment, typ = AMSentence.split_supertag(self.vocab.get_token_from_index(top_k, namespace="supertag_labels"))
                    score = supertag_scores[i,word,top_k]
                    supertags_for_this_word.append((score, fragment, typ))
                if bot_id not in top_k_supertags[i,word]: #\bot is not in the top k, but we have to add it anyway in order for the decoder to work properly.
                    fragment,typ = AMSentence.split_supertag(AMSentence.get_bottom_supertag())
                    supertags_for_this_word.append((supertag_scores[i,word,bot_id],fragment,typ))
                supertags_for_this_sentence.append(supertags_for_this_word)
            all_supertags.append(supertags_for_this_sentence)
            all_predicted_lex_labels.append([self.vocab.get_token_from_index(label,namespace="lex_labels") for label in lexlabels[i,1:length]])
            head_indices.append(instance_heads_cpu)

        output_dict["lexlabels"] = all_predicted_lex_labels
        output_dict["supertags"] = all_supertags
        output_dict["root"] = roots
        output_dict["label_logits"] = all_edge_label_logits
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
        r["Supertagging Acc"] = self._supertagging_acc.get_metric(reset)
        r["Lex Label Acc"] = self._lexlabel_acc.get_metric(reset)

        if reset: #epoch done
            if self.training: #done on the training data
                self.current_epoch += 1
            else: #done on dev/test data
                if self.validation_evaluator:
                    metrics = self.validation_evaluator.eval(self, self.current_epoch)
                    for name, val in metrics.items():
                        r[name] = val
        return r
