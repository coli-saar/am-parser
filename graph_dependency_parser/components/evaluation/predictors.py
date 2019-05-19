from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Tuple

from allennlp.common import Params, Registrable
from allennlp.data import DatasetReader, DataIterator
from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.models import Model
from allennlp.common.util import lazy_groups_of

from graph_dependency_parser.am_algebra.label_decoder import AMDecoder
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
        Creates a predictor from a DatasetReader, optionally takes an AllenNLP model. The model can also be given later using set_model.
        If evaluation is required, en evaluation_command can be supplied as well.
        :param dataset_reader:
        :param evaluation_command:
        :param model:
        """
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

    def parse_and_save(self, input_file : str, output_file: str) -> None:
        """
        Parses an input file and saves it to some given output file. Old content will be overwritten.
        :param input_file:
        :param output_file:
        :return:
        """
        raise NotImplementedError()

    def parse_and_eval(self, input_file : str, gold_file: str) -> Dict[str,float]:
        """
        Given an input file and a gold standard file, parses the input, saves the output in a temporary directory
        and calls the evaluation command
        :param input_file:
        :param gold_file:
        :return: a dictionary with evaluation metrics as delivered by evaluation_command
        """
        assert self.evaluation_command, "parse_and_eval needs evaluation_command to be given"
        with tempfile.TemporaryDirectory() as tmpdirname:
            filname = tmpdirname+"/prediction"
            self.parse_and_save(input_file,filname)
            return self.evaluation_command.evaluate(filname,gold_file)

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
        return cls(DatasetReader.from_params(params["dataset_reader"]),model=model)


def dict_to_conllu(d : Dict[str,list]) -> str:
    """
    Converts dict to conllu string.
    :param d: output of decode() of GraphDependencyParser
    :return: a string with its conllu representation
    """
    output = []
    for i, (w,  head, edge) in enumerate(zip(d["words"],d["predicted_heads"],d["predicted_labels"]),1):
        line = [i, w, "_","_","_","_",head,edge,str(head)+":"+edge,"_"]
        output.append("\t".join(str(x) for x in line))
    return "\n".join(output)


@Predictor.register("conllu_predictor")
class ConlluPredictor(Predictor):
    """
    Predictor that specifically generates conllu output.
    """

    def parse(self, input_file: str) -> List[Dict]:
        """
        Loads an input_file and returns a list of predictions from the model's forward_on_instances method
        :param input_file:
        :return:
        """
        assert self.model, "model must be given, either to the constructor or to set_model"
        instances = self.dataset_reader.read(input_file)
        prev_training_status = self.model.training
        self.model.train(False)
        predictions = self.dataset_reader.restore_order(forward_on_instances(self.model, instances,self.data_iterator))
        self.model.train(prev_training_status) #reset training status to whatever it was before
        return predictions

    def parse_and_save(self, input_file : str, output_file: str) -> None:
        """
        Parses an input file and saves it to some given output file. Old content will be overwritten.
        :param input_file:
        :param output_file:
        :return:
        """
        predictions = self.parse(input_file)
        with open(output_file, "w") as f:
            for p in predictions:
                f.write(dict_to_conllu(p))
                f.write("\n\n")



@Predictor.register("amconll_predictor")
class AMconllPredictor(Predictor):
    """
    Predictor that calls the fixed-tree decoder.
    """

    def __init__(self, dataset_reader: DatasetReader, k:int,give_up:float, threads:int = 4,data_iterator: DataIterator = None,
                 evaluation_command: BaseEvaluationCommand = None, model: Model = None, batch_size: int = 64):
        """
        Creates a predictor from a DatasetReader, optionally takes an AllenNLP model. The model can also be given later using set_model.
        If evaluation is required, en evaluation_command can be supplied as well.
        :param dataset_reader:
        :param k: number of supertags to be used during decoding
        :param give_up: time limit in seconds before retry parsing with k-1 supertags
        :param threads: number of parallel threads to parse corpus
        :param evaluation_command:
        :param model:
        """
        super().__init__(dataset_reader,data_iterator,evaluation_command,model,batch_size)
        self.k = k
        self.threads = threads
        self.give_up = give_up

    def parse_and_save(self, input_file : str, output_file: str) -> None:
        """
        Parses an input file and saves it to some given output file. Old content will be overwritten.
        :param input_file:
        :param output_file:
        :return:
        """
        assert self.model, "model must be given, either to the constructor or to set_model"
        instances = self.dataset_reader.read(input_file)
        prev_training_status = self.model.training
        self.model.train(False)
        predictions = self.dataset_reader.restore_order(forward_on_instances(self.model, instances,self.data_iterator))
        self.model.train(prev_training_status) #reset training status to whatever it was before
        i2edge_label = [ self.model.vocab.get_token_from_index(i,namespace="head_tags") for i in range(self.model.vocab.get_vocab_size("head_tags"))]
        decoder = AMDecoder(output_file,i2edge_label)
        for pred in predictions:
            attributes = pred["attributes"]
            am_sentence = AMSentence(pred["words"],attributes) #(form,replacement,lemma,pos,ne)
            sentence = list(zip(am_sentence.get_tokens(shadow_art_root=False),am_sentence.get_replacements(), am_sentence.get_lemmas(), am_sentence.get_pos(), am_sentence.get_ner()))
            decoder.add_sentence(pred["root"],pred["predicted_heads"],pred["label_logits"],pred["lexlabels"],pred["supertags"], sentence, am_sentence.attributes_to_list())
        decoder.decode(self.threads,self.k,self.give_up)

        return predictions




class ValidationEvaluator(Registrable):
    """
    A wrapper around a predictor that remembers system input and gold file
    Intended for easy use in configuration files.
    """
    def __init__(self, system_input : str, gold_file : str, predictor : Predictor, use_from_epoch: int = 1) -> None:
        self.system_input = system_input
        self.gold_file = gold_file
        self.predictor = predictor
        self.use_from_epoch = use_from_epoch

    def eval(self,model, epoch) -> Dict[str, float]:
        if epoch < self.use_from_epoch:
            return dict()
        self.predictor.set_model(model)
        return self.predictor.parse_and_eval(self.system_input, self.gold_file)
