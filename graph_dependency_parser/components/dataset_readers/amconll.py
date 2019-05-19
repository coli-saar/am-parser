from typing import Dict, Tuple, List, Any
import logging

from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from graph_dependency_parser.components.dataset_readers.amconll_tools import parse_amconll, AMSentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("amconll")
class AMConllDatasetReader(DatasetReader):
    """
    Reads a file in amconll format containing AM dependency trees.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as amconll_file:
            logger.info("Reading AM dependency trees from amconll dataset at: %s", file_path)

            for i,am_sentence in  enumerate(parse_amconll(amconll_file)):
                yield self.text_to_instance(i,am_sentence)

    @overrides
    def text_to_instance(self,  # type: ignore
                         position_in_corpus : int,
                         am_sentence: AMSentence) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        position_in_corpus : ``int``, required.
            The index of this sentence in the corpus.
        am_sentence : ``AMSentence``, required.
            The words in the sentence to be encoded.

        Returns
        -------
        An instance containing words, pos tags, dependency edge labels, head
        indices, supertags and lexical labels as fields.
        """
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in am_sentence.get_tokens(shadow_art_root=True)], self._token_indexers)
        fields["words"] = tokens
        fields["pos_tags"] = SequenceLabelField(am_sentence.get_pos(), tokens, label_namespace="pos")
        fields["ner_tags"] = SequenceLabelField(am_sentence.get_ner(), tokens, label_namespace="ner_labels")
        fields["lemmas"] = SequenceLabelField(am_sentence.get_lemmas(), tokens, label_namespace="lemmas")
        fields["supertags"] = SequenceLabelField(am_sentence.get_supertags(), tokens, label_namespace="supertag_labels")
        fields["lexlabels"] = SequenceLabelField(am_sentence.get_lexlabels(), tokens, label_namespace="lex_labels")
        fields["head_tags"] = SequenceLabelField(am_sentence.get_edge_labels(),tokens, label_namespace="head_tags")
        fields["head_indices"] = SequenceLabelField(am_sentence.get_heads(),tokens,label_namespace="head_index_tags")

        fields["metadata"] = MetadataField({"words": am_sentence.words, "attributes": am_sentence.attributes, "position_in_corpus" : position_in_corpus})
        return Instance(fields)

    @staticmethod
    def restore_order(instances : List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tries to restore the order that was used when the instances were read.
        :param instances:
        :return:
        """
        return sorted(instances, key=lambda d: d["position_in_corpus"])
