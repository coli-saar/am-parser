from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch
from allennlp.common import Registrable

from topdown_parser.am_algebra import AMType
from topdown_parser.dataset_readers.additional_lexicon import AdditionalLexicon, Lexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.edge_label_model import EdgeLabelModel
from topdown_parser.nn.utils import get_device_id
from topdown_parser.transition_systems.batched_parsing_state import BatchedParsingState
from topdown_parser.transition_systems.decision import Decision, DecisionBatch
from topdown_parser.transition_systems.parsing_state import ParsingState




class TransitionSystem(Registrable):

    def __init__(self, additional_lexicon : AdditionalLexicon):
        self.additional_lexicon = additional_lexicon

    def guarantees_well_typedness(self) -> bool:
        raise NotImplementedError()

    def get_unconstrained_version(self) -> "TransitionSystem":
        """
        Return an unconstrained version that does not do type checking.
        :return:
        """
        raise NotImplementedError()

    def is_on_gpu(self):
        """
        Is this object a GPUTransitionSystem?
        :return:
        """
        return False

    def validate_model(self, parser : "TopDownDependencyParser") -> None:
        """
        Check if the parsing model produces all the scores that we need.
        :param parser:
        :return:
        """
        return

    def get_order(self, sentence : AMSentence) -> Iterable[Decision]:
        """
        Pre-compute the sequence of decisions that parser should produce.
        The decisions use 1-based indexing for nodes.
        :param sentence:
        :return:
        """
        raise NotImplementedError()

    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        """
        Check if the predicted sentence is exactly the same as the gold sentence.
        Has to be implemented because not all transition system have to predict lexical as-graphs.
        :param gold_sentence:
        :param predicted:
        :return:
        """
        raise NotImplementedError()

    def initial_state(self, sentence : AMSentence, decoder_state : Any) -> ParsingState:
        raise NotImplementedError()

    def step(self, state: ParsingState, decision : Decision, in_place : bool = False) -> ParsingState:
        """
        Applies a decision to a parsing state, yielding a new parsing state.
        :param state:
        :param decision:
        :return:
        """
        raise NotImplementedError()

    def make_decision(self, scores: Dict[str, torch.Tensor], state : ParsingState) -> Decision:
        """
        Informs the transition system about the last node chosen
        Returns the index of the node that will get a child next according to the transitions system.
        :param scores: additional choices for each batch element, like edge labels for example, contains edge existence scores.
        :return: a tensor of shape (batch_size,) of currently active nodes
            and a tensor of shape (batch_size, input_seq_len) which for every input position says if it is a valid next choice.
            input_seq_len is set above in reset_parses
        """
        raise NotImplementedError()

    def top_k_decision(self, scores: Dict[str, torch.Tensor], state: ParsingState, k : int) -> List[Decision]:
        raise NotImplementedError()

    def assumes_greedy_ok(self) -> Set[str]:
        """
        The dictionary keys of the context provider which we make greedy decisions on in top_k_decisions
        because we assume these choices won't impact future scores.
        TODO something's not right in terms of hierarchy: gather_context should be at level of ParsingState and not CommonParsingState?
        :return:
        """
        raise NotImplementedError()

    def predict_supertag_from_tos(self) -> bool:
        """
        shall we try to predict the supertag using the encoding of the nodes on top of the stack? (dfs-children-first)
        or shall we try to predict it from the nodes just selected? (dfs)
        :return:
        """
        raise NotImplementedError()

    def decision_to_score(self, sentence : AMSentence, decision) -> Dict[str, torch.Tensor]:
        """
        In order to simulate scores for training data.
        :param sentence:
        :param decision:
        :return:
        """
        children_scores = torch.zeros(len(sentence)+1)
        children_scores[decision.position] = 1
        constant_scores = torch.zeros(self.additional_lexicon.vocab_size("constants"))
        if decision.supertag is not None:
            constant_scores[self.additional_lexicon.get_id("constants", "--TYPE--".join(decision.supertag))] = 1

        term_type_scores = torch.zeros(self.additional_lexicon.vocab_size("term_types"))
        if decision.termtyp is not None:
            term_type_scores[self.additional_lexicon.get_id("term_types", str(decision.termtyp))] = 1

        lex_label_scores = torch.zeros(self.additional_lexicon.vocab_size("lex_labels"))
        if decision.lexlabel != "":
            lex_label_scores[self.additional_lexicon.get_id("lex_labels", decision.lexlabel)] = 1

        edge_label_scores = torch.zeros(len(sentence)+1, self.additional_lexicon.vocab_size("edge_labels"))
        if decision.label != "":
            edge_label_scores[decision.position, self.additional_lexicon.get_id("edge_labels", decision.label)] = 1

        return {"children_scores": children_scores, "constants_scores": constant_scores,
                "term_types_scores": term_type_scores, "lex_labels_scores" : lex_label_scores,
                "lex_labels" : torch.randint(0, self.additional_lexicon.vocab_size("lex_labels"), (1,)),
                "all_labels_scores" : edge_label_scores}

    def fuzz_scores(self, sentence: AMSentence, beam_search: bool) -> Dict[str, torch.Tensor]:
        children_scores = torch.rand(len(sentence)+1)
        constant_scores = torch.rand(self.additional_lexicon.vocab_size("constants"))

        term_type_scores = torch.rand(self.additional_lexicon.vocab_size("term_types"))

        lex_label_scores = torch.rand(self.additional_lexicon.vocab_size("lex_labels"))

        r = {"children_scores": children_scores, "constants_scores": constant_scores,
             "lex_labels" : torch.randint(0, self.additional_lexicon.vocab_size("lex_labels"), (1,)),
                "term_types_scores": term_type_scores, "lex_labels_scores" : lex_label_scores}

        r["all_labels_scores"] = torch.rand((len(sentence)+1, self.additional_lexicon.vocab_size("edge_labels")))

        return r


    ### STUFF FOR PARSING ON GPU ####
    def prepare(self, device : Optional[int]) -> None:
        """
        Move pre-computed arrays for type checking on GPU to a specific device.
        :param device:
        :return:
        """
        return

    def gpu_initial_state(self, sentences: List[AMSentence], decoder_state: Any, device: Optional[int] = None) -> BatchedParsingState:
        """
        Get initial state for a list of sentences and given object that represents the decoder state for all sentences.
        :param sentences:
        :param decoder_state:
        :param device:
        :return:
        """
        raise NotImplementedError()

    def gpu_step(self, state: BatchedParsingState, decision_batch: DecisionBatch) -> None:
        """
        Applies a decision to a parsing state.
        :param state:
        :param decision_batch:
        :return:
        """
        raise NotImplementedError()

    def gpu_make_decision(self, scores: Dict[str, torch.Tensor], state : BatchedParsingState) -> DecisionBatch:
        """
        Informs the transition system about the last node chosen
        Returns the index of the node that will get a child next according to the transitions system.
        :param scores: additional choices for each batch element, like edge labels for example, contains edge existence scores.
        :return: a tensor of shape (batch_size,) of currently active nodes
            and a tensor of shape (batch_size, input_seq_len) which for every input position says if it is a valid next choice.
            input_seq_len is set above in reset_parses
        """
        raise NotImplementedError()

    def gpu_decision_to_score(self, sentence : AMSentence, decision) -> Dict[str, torch.Tensor]:
        """
        In order to simulate scores for training data.
        :param sentence:
        :param decision:
        :return:
        """
        children_scores = torch.zeros(1, len(sentence)+1)
        children_scores[0, decision.position] = 1
        constant_scores = torch.zeros(1, self.additional_lexicon.vocab_size("constants"))
        if decision.supertag is not None:
            constant_scores[0, self.additional_lexicon.get_id("constants", "--TYPE--".join(decision.supertag))] = 1

        term_type_scores = torch.zeros(1, self.additional_lexicon.vocab_size("term_types"))
        if decision.termtyp is not None:
            term_type_scores[0, self.additional_lexicon.get_id("term_types", str(decision.termtyp))] = 1

        lex_label_scores = torch.zeros(1, self.additional_lexicon.vocab_size("lex_labels"))
        if decision.lexlabel != "":
            lex_label_scores[0, self.additional_lexicon.get_id("lex_labels", decision.lexlabel)] = 1

        edge_label_scores = torch.zeros(1, len(sentence)+1, self.additional_lexicon.vocab_size("edge_labels"))
        if decision.label != "":
            edge_label_scores[0, decision.position, self.additional_lexicon.get_id("edge_labels", decision.label)] = 1

        return {"children_scores": children_scores, "constants_scores": constant_scores,
                "term_types_scores": term_type_scores, "lex_labels_scores" : lex_label_scores,
                "lex_labels" : torch.tensor([self.additional_lexicon.get_id("lex_labels", decision.lexlabel)]),
                "all_labels_scores" : edge_label_scores}

    def gpu_fuzz_scores(self, sentence: AMSentence) -> Dict[str, torch.Tensor]:
        children_scores = torch.rand(1, len(sentence)+1)
        constant_scores = torch.rand(1, self.additional_lexicon.vocab_size("constants"))

        term_type_scores = torch.rand(1, self.additional_lexicon.vocab_size("term_types"))

        lex_label_scores = torch.rand(1, self.additional_lexicon.vocab_size("lex_labels"))

        r = {"children_scores": children_scores, "constants_scores": constant_scores,
             "lex_labels" : torch.randint(0, self.additional_lexicon.vocab_size("lex_labels"), (1,)),
             "term_types_scores": term_type_scores, "lex_labels_scores" : lex_label_scores}

        r["all_labels_scores"] = torch.rand((1, len(sentence)+1, self.additional_lexicon.vocab_size("edge_labels")))

        return r

