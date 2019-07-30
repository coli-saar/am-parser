import os
from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Tuple, Union

from allennlp.common import Params, Registrable
from allennlp.data import DatasetReader, DataIterator
from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.models import Model
from allennlp.common.util import lazy_groups_of

from graph_dependency_parser.am_algebra.label_decoder import AMDecoder
from graph_dependency_parser.components.dataset_readers.amconll import AMConllDatasetReader
from graph_dependency_parser.components.dataset_readers.amconll_tools import AMSentence
from graph_dependency_parser.components.evaluation.commands import BaseEvaluationCommand
from graph_dependency_parser.components.evaluation.iterator import forward_on_instances

import tempfile


class Predictor (Registrable):
    """
    A class that can make predictions for an input file. Not to be confused with AllenNLP's own predictors.
    """
    def __init__(self, dataset_reader : DatasetReader, data_iterator: DataIterator = None ,evaluation_command : BaseEvaluationCommand = None, model : Model = None, batch_size:int = 64):
        """
        Creates a predictor from an AMConllDatasetReader, optionally takes an AllenNLP model. The model can also be given later using set_model.
        If evaluation is required, en evaluation_command can be supplied as well.
        :param dataset_reader: an AMConllDatasetReader
        :param evaluation_command:
        :param model:
        """
        assert isinstance(dataset_reader, AMConllDatasetReader), "A predictor in the am-parser must take an AMConllDatasetReader"
        self.dataset_reader = dataset_reader
        self.model = model
        self.evaluation_command = evaluation_command
        self.batch_size = batch_size
        if data_iterator is None:
            self.data_iterator = BasicIterator()
        else:
            self.data_iterator = data_iterator

    def set_model(self, model : Model):
        self.model = model

    def set_evaluation_command(self, evaluation_command : BaseEvaluationCommand):
        self.evaluation_command = evaluation_command

    def parse_and_save(self, formalism: str, input_file : str, output_file: str) -> None:
        """
        Parses an input file and saves it to some given output file. Old content will be overwritten.
        :param input_file:
        :param formalism: the name of the formalism of the input_file
        :param output_file:
        :return:
        """
        raise NotImplementedError()

    def parse_and_eval(self, formalism:str, input_file : str, gold_file: str, filename :Union[str, None]) -> Dict[str,float]:
        """
        Given an input file and a gold standard file, parses the input, saves the output in a temporary directory
        and calls the evaluation command
        :param input_file:
        :param formalism: the name of the formalism of the input_file
        :param gold_file:
        :return: a dictionary with evaluation metrics as delivered by evaluation_command
        """
        assert self.evaluation_command, "parse_and_eval needs evaluation_command to be given"
        if not filename:
            with tempfile.TemporaryDirectory() as tmpdirname:
                filename = tmpdirname+"/prediction"
                self.parse_and_save(formalism, input_file, filename)
                return self.evaluation_command.evaluate(filename,gold_file)
        else:
            self.parse_and_save(formalism, input_file, filename)
            return self.evaluation_command.evaluate(filename, gold_file)

    @classmethod
    def from_config(cls, config_file : str, serialization_dir : str) -> "Predictor":
        """
        Creates a predictor from a configuration file (jsonnet) and a model directory.
        :param config_file:
        :param serialization_dir:
        :return:
        """
        params = Params.from_file(config_file)
        model = Model.load(params, serialization_dir)
        dr = DatasetReader.from_params(params["dataset_reader"])
        assert isinstance(dr, AMConllDatasetReader), "A predictor in the am-parser must take an AMConllDatasetReader"
        return cls(dr,model=model)


@Predictor.register("amconll_predictor")
class AMconllPredictor(Predictor):
    """
    Predictor that calls the fixed-tree decoder.
    """

    def __init__(self, dataset_reader: DatasetReader, k:int,give_up:float, threads:int = 4,data_iterator: DataIterator = None,
                 evaluation_command: BaseEvaluationCommand = None, model: Model = None, batch_size: int = 64, give_up_k_1 : float = None):
        """
        Creates a predictor from an AMConllDatasetReader, optionally takes an AllenNLP model. The model can also be given later using set_model.
        If evaluation is required, en evaluation_command can be supplied as well.
        :param dataset_reader: an AMConllDatasetReader
        :param k: number of supertags to be used during decoding
        :param give_up: time limit in seconds before retry parsing with k-1 supertags
        :param threads: number of parallel threads to parse corpus
        :param give_up_k_1: if given, how long to wait before skipping sentence entirely ("back off" from k=1 to k=0)
        :param evaluation_command:
        :param model:
        """
        super().__init__(dataset_reader,data_iterator,evaluation_command,model,batch_size)
        self.k = k
        self.threads = threads
        self.give_up = give_up
        if give_up_k_1 is None:
            self.give_up_k_1 = give_up
        else:
            self.give_up_k_1 = give_up_k_1

    def parse_and_save(self, formalism : str, input_file : str, output_file: str) -> None:
        """
        Parses an input file and saves it to some given output file. Old content will be overwritten.
        :param input_file:
        :param formalism: the name of the formalism of the input_file
        :param output_file:
        :return:
        """
        assert self.model, "model must be given, either to the constructor or to set_model"
        instances = self.dataset_reader.read([[formalism, input_file]]) #we need to give the formalism to amconll dataset_reader
        prev_training_status = self.model.training
        self.model.train(False)
        predictions = self.dataset_reader.restore_order(forward_on_instances(self.model, instances,self.data_iterator))
        self.model.train(prev_training_status) #reset training status to whatever it was before
        i2edge_label = [ self.model.vocab.get_token_from_index(i,namespace=formalism+"_head_tags") for i in range(self.model.vocab.get_vocab_size(formalism+"_head_tags"))]
        decoder = AMDecoder(output_file,i2edge_label)
        for pred in predictions:
            attributes = pred["attributes"]
            am_sentence = AMSentence(pred["words"],attributes) #(form,replacement,lemma,pos,ne)
            sentence = list(zip(am_sentence.get_tokens(shadow_art_root=False),am_sentence.get_replacements(), am_sentence.get_lemmas(), am_sentence.get_pos(), am_sentence.get_ner(), am_sentence.get_ranges()))
            decoder.add_sentence(pred["root"],pred["predicted_heads"],pred["label_logits"],pred["lexlabels"],pred["supertags"], sentence, am_sentence.attributes_to_list())
        decoder.decode(self.threads,self.k,self.give_up,self.give_up_k_1)


class Evaluator(Registrable):
    """
    For use in configuration files. Abstract class that only defines what an evaluator should look like.
    """
    def eval(self, model, epoch,model_path=None) -> Dict[str,float]:
        raise NotImplementedError()


@Evaluator.register("standard_evaluator")
class StandardEvaluator(Evaluator):
    """
    A wrapper around a predictor that remembers system input and gold file
    Intended for easy use in configuration files.
    """
    def __init__(self, formalism:str, system_input : str, gold_file : str, predictor : Predictor, use_from_epoch: int = 1) -> None:
        self.formalism = formalism
        self.system_input = system_input
        self.gold_file = gold_file
        self.predictor = predictor
        self.use_from_epoch = use_from_epoch

    def eval(self,model, epoch, model_path=None) -> Dict[str, float]:
        if epoch < self.use_from_epoch:
            return dict()
        self.predictor.set_model(model)
        if model_path:
            filename = model_path + "/" + "dev_epoch_"+str(epoch)+".amconll"
            return self.predictor.parse_and_eval(self.formalism, self.system_input, self.gold_file, filename=filename)
        else: #use temporary directory
            return self.predictor.parse_and_eval(self.formalism, self.system_input, self.gold_file,None)


@Evaluator.register("dummy_evaluator")
class DummyEvaluator(Evaluator):

    def eval(self,model, epoch,model_path=None) -> Dict[str, float]:
        return dict()

@Evaluator.register("empty_mrp_evaluator")
class EmptyMRPEvaluator(Evaluator):
    """
    A wrapper around a predictor that remembers system input.
    """
    def __init__(self, formalism:str, system_input : str, predictor : Predictor, postprocessing : List[str]) -> None:
        """

        :param formalism:
        :param system_input:
        :param predictor:
        :param postprocessing: a list of strings with postprocessing commands, you can use {system_output} as a placeholder
        """
        self.postprocessing = postprocessing
        self.formalism = formalism
        self.system_input = system_input
        self.predictor = predictor

    def eval(self,model, epoch, model_path=None) -> Dict[str, float]:
        self.predictor.set_model(model)
        if model_path:
            filename = model_path + "/" + "test_"+str(self.formalism)+".amconll"
            self.predictor.parse_and_save(self.formalism, self.system_input, filename)
            for cmd in self.postprocessing:
                cmd = cmd.format(system_output=filename)
                os.system(cmd)
            return dict()
        else: #use temporary directory
            raise ValueError("Need to get model_path!")
