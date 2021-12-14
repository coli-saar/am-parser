import time
from typing import Dict, Iterable, List, Optional
import logging

import torch
from allenpipeline import OrderedDatasetReader
from overrides import overrides

import multiprocessing as mp

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, ArrayField, ListField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from .context_field import ContextField
from .amconll_tools import parse_amconll, AMSentence
from ..am_algebra.tools import is_welltyped

from tqdm import tqdm

from ..transition_systems.parsing_state import ParsingState
from ..transition_systems.transition_system import TransitionSystem

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("amconll")
class AMConllDatasetReader(OrderedDatasetReader):
    """
    Reads a file in amconll format containing AM dependency trees.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    """

    def __init__(self,
                 transition_system: TransitionSystem,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False, fraction: float = 1.0,
                 overwrite_formalism : str = None,
                 workers : int = 1,
                 run_oracle : bool = True,
                 device : int = None,
                 fuzz: bool = False,
                 fuzz_beam_search : bool = False,
                 use_tqdm : bool = False,
                 only_read_fraction_if_train_in_filename: bool = False,
                 read_tokens_only: bool = False) -> None:
        super().__init__(lazy)
        self.read_tokens_only = read_tokens_only
        self.use_tqdm = use_tqdm
        self.fuzz_beam_search = fuzz_beam_search
        self.fuzz = fuzz
        self.run_oracle = run_oracle
        self.workers = workers
        self.overwrite_formalism = overwrite_formalism
        self.transition_system: TransitionSystem = transition_system
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.fraction = fraction
        self.only_read_fraction_if_train_in_filename = only_read_fraction_if_train_in_filename
        self.lexicon = transition_system.additional_lexicon
        if device is not None and device < 0:
            device = None
        self.device = device

    def collect_sentences(self, file_path : str) -> List[AMSentence]:
        file_path = cached_path(file_path)
        if self.fraction < 0.9999 and (not self.only_read_fraction_if_train_in_filename or (
                self.only_read_fraction_if_train_in_filename and "train" in file_path)):
            with open(file_path, 'r') as amconll_file:
                logger.info("Reading a fraction of " + str(
                    self.fraction) + " of the AM dependency trees from amconll dataset at: %s", file_path)
                sents = list(parse_amconll(amconll_file))
                return sents[:int(len(sents)*self.fraction)]
        else:
            with open(file_path, 'r') as amconll_file:
                logger.info("Reading AM dependency trees from amconll dataset at: %s", file_path)
                return list(parse_amconll(amconll_file))

    def read_file(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        sents : List[AMSentence] = self.collect_sentences(file_path)
        t1 = time.time()
        if self.workers < 2: #or self.workers >= len(sents):
            #import cProfile
            #with cProfile.Profile() as pr:
            if self.use_tqdm:
                sents = tqdm(sents)
            r = [self.text_to_instance(s) for s in sents]
            #pr.print_stats()
        else:
            with mp.Pool(self.workers) as pool:
                r = pool.map(self.text_to_instance, sents)
        delta = time.time() - t1
        logger.info(f"Reading took {round(delta,3)} seconds")
        return [x for x in r if x is not None]

    @overrides
    def text_to_instance(self,  # type: ignore
                         am_sentence: AMSentence) -> Optional[Instance]:
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

        if self.overwrite_formalism is not None:
            formalism = self.overwrite_formalism
        else:
            formalism = am_sentence.attributes["framework"]

        am_sentence = am_sentence.fix_dev_edge_labels()

        tokens = TextField([Token(w) for w in am_sentence.get_tokens(shadow_art_root=True)], self._token_indexers)
        fields["words"] = tokens
        fields["pos_tags"] = SequenceLabelField(am_sentence.get_pos(), tokens, label_namespace="pos")
        fields["ner_tags"] = SequenceLabelField(am_sentence.get_ner(), tokens, label_namespace="ner")
        fields["lemmas"] = SequenceLabelField(am_sentence.get_lemmas(), tokens, label_namespace="lemmas")
        fields["metadata"] = MetadataField({"formalism": formalism,
                                            "am_sentence": am_sentence,
                                            "is_annotated": am_sentence.is_annotated()})
        if not am_sentence.is_annotated() or self.read_tokens_only:
            return Instance(fields)

        #We are dealing with training data, prepare it accordingly.

        if not is_welltyped(am_sentence):
            print("Skipping non-well-typed AMDep tree.")
            print(am_sentence.get_tokens(shadow_art_root=False))
            return None

        am_sentence = am_sentence.normalize_types()

        # print(am_sentence.get_tokens(False))
        # print(am_sentence.get_heads())
        # print(am_sentence.get_edge_labels())
        # print([ AMSentence.split_supertag(s)[1] for s in am_sentence.get_supertags()])

        decisions = list(self.transition_system.get_order(am_sentence))

        ##################################################################
        #Validate decision sequence and gather context:
        assert decisions[0].position == 0
        #assert active_nodes[0] == 0

        #assert len(decisions) == len(active_nodes) + 1

        # Try to reconstruct tree
        stripped_sentence = am_sentence.strip_annotation()

        state : ParsingState = self.transition_system.initial_state(stripped_sentence, None)
        active_nodes = [0]

        contexts = dict()
        for i in range(1, len(decisions)):
            #Gather context
            context = state.gather_context(None) #device: cpu
            for k, v in context.items():
                if k not in contexts:
                    contexts[k] = []
                contexts[k].append(v.cpu().numpy())

            state = self.transition_system.step(state, decisions[i], in_place=True)

            if i == len(decisions) - 1:  # there are no further active nodes after this step
                break

            active_nodes.append(state.active_node)

        assert state.is_complete(), f"State should be complete: {state}"
        reconstructed = state.extract_tree()

        assert self.transition_system.check_correct(am_sentence, reconstructed), f"Could not reconstruct this sentence\n: {am_sentence.get_tokens(False)}"

        if self.run_oracle:
            ## Now with oracle scores:
            stripped_sentence = am_sentence.strip_annotation()

            state : ParsingState = self.transition_system.initial_state(stripped_sentence, None)
            for decision in decisions[1:]:
                scores = self.transition_system.decision_to_score(stripped_sentence, decision)
                decision_prime = self.transition_system.make_decision(scores, state)
                state = self.transition_system.step(state, decision_prime, in_place=True)
            assert state.is_complete()
            reconstructed = state.extract_tree()
            assert self.transition_system.check_correct(am_sentence, reconstructed), f"Could not reconstruct this sentence\n: {am_sentence.get_tokens(False)}"

        if self.fuzz:
            # Now fuzz
            rng_state = torch.random.get_rng_state()
            stripped_sentence = am_sentence.strip_annotation()
            # print([w.token for w in am_sentence.words])
            hash_value = len(am_sentence) + sum(len(w.token)*ord(w.token[0])*ord(w.token[-1]) for w in am_sentence.words)
            # print(hash_value)
            torch.random.manual_seed(hash_value)
            state : ParsingState = self.transition_system.initial_state(stripped_sentence, None)
            for _ in range(2*len(stripped_sentence)+1):
                scores = self.transition_system.fuzz_scores(stripped_sentence, beam_search=False)
                decision = self.transition_system.make_decision(scores, state)
                state = self.transition_system.step(state, decision, in_place=True)

            assert state.is_complete()
            assert (not self.transition_system.guarantees_well_typedness()) or is_welltyped(state.extract_tree())
            torch.random.set_rng_state(rng_state)

        if self.fuzz_beam_search:
            rng_state = torch.random.get_rng_state()
            stripped_sentence = am_sentence.strip_annotation()
            hash_value = len(am_sentence) + sum(len(w.token)*ord(w.token[0])*ord(w.token[-1]) for w in am_sentence.words)
            torch.random.manual_seed(hash_value)
            state : ParsingState = self.transition_system.initial_state(stripped_sentence, None)
            for _ in range(2*len(stripped_sentence)+1):
                scores = self.transition_system.fuzz_scores(stripped_sentence, beam_search=True)
                decision = self.transition_system.top_k_decision(scores, state, k=4)[0]
                state = self.transition_system.step(state, decision, in_place=True)

            assert state.is_complete()
            assert (not self.transition_system.guarantees_well_typedness()) or is_welltyped(state.extract_tree())
            torch.random.set_rng_state(rng_state)

        ##################################################################

        # Create instance
        seq = ListField([LabelField(decision.position, skip_indexing=True) for decision in decisions])
        fields["seq"] = seq
        fields["active_nodes"] = ListField(
            [LabelField(active_node, skip_indexing=True) for active_node in active_nodes])

        fields["labels"] = SequenceLabelField([self.lexicon.get_id("edge_labels",decision.label) for decision in decisions], seq)
        fields["label_mask"] = SequenceLabelField([int(decision.label != "") for decision in decisions], seq)

        fields["term_types"] = SequenceLabelField([self.lexicon.get_id("term_types", str(decision.termtyp)) if decision.termtyp is not None else 0 for decision in decisions], seq)

        fields["term_type_mask"] = SequenceLabelField([int(decision.termtyp is not None) for decision in decisions], seq)

        fields["lex_labels"] = SequenceLabelField([self.lexicon.get_id("lex_labels", decision.lexlabel) for decision in decisions], seq)
        fields["lex_label_mask"] = SequenceLabelField([int(decision.lexlabel != "") for decision in decisions], seq)

        fields["supertags"] = SequenceLabelField([self.lexicon.get_id("constants","--TYPE--".join(decision.supertag)) for decision in decisions], seq)

        fields["supertag_mask"] = SequenceLabelField([int(decision.supertag[1] != "") for decision in decisions], seq)

        fields["heads"] = SequenceLabelField(am_sentence.get_heads(), tokens)

        fields["context"] = ContextField(
            {name: ListField([ArrayField(array, dtype=array.dtype) for array in liste]) for name, liste in
             contexts.items()})

        # fields["supertags"] = SequenceLabelField(am_sentence.get_supertags(), tokens, label_namespace=formalism+"_supertag_labels")
        # fields["lexlabels"] = SequenceLabelField(am_sentence.get_lexlabels(), tokens, label_namespace=formalism+"_lex_labels")
        # fields["head_tags"] = SequenceLabelField(am_sentence.get_edge_labels(),tokens, label_namespace=formalism+"_head_tags") #edge labels
        # fields["head_indices"] = SequenceLabelField(am_sentence.get_heads(),tokens,label_namespace="head_index_tags")

        return Instance(fields)
