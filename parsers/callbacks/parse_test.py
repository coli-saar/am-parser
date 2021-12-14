import os
from typing import Optional, List

from allennlp.common.checks import ConfigurationError
from allennlp.models import Model
from allenpipeline import Annotator, BaseEvaluationCommand
from allenpipeline.callback import Callback
from comet_ml import Experiment

@Callback.register("parse-test")
class ParseTest(Callback):

    def __init__(self, test_commands : List[BaseEvaluationCommand], system_inputs : List[str], names : List[str], active : bool = True):
        """
        Parses test sets and performs evaluation.
        :param test_commands: test commands, pairing up with system inputs
        :param system_inputs: test files
        :param names: name for annotated data, e.g. names=["DM_id","DM_ood"] will result in test_DM_iid.txt and test_DM_ood.txt
        the metrics reported will then be DM_id_F etc.
        """
        self.active = active
        assert isinstance(test_commands, list)
        assert isinstance(system_inputs, list)
        assert isinstance(names, list)
        assert len(test_commands) == len(system_inputs)
        assert len(names) == len(system_inputs)

        self.names = names
        self.system_inputs = system_inputs
        self.test_commands = test_commands

        for input in self.system_inputs:
            assert os.path.exists(input), f"{input} does not exist"

        for command in test_commands:
            if command.gold_file is None:
                raise ConfigurationError("Callback requires a gold file to be set.")

    def call(self, annotator : Annotator, model : Model, trainer : Optional["PipelineTrainer"] = None, experiment : Optional[Experiment] = None):
        if self.active:
            for i in range(len(self.system_inputs)):
                filename = trainer._serialization_dir+f"/test_{self.names[i]}.txt"
                annotator.annotate_file(model, self.system_inputs[i], filename)
                results = self.test_commands[i].evaluate(filename)
                trainer.metrics.update({"test_"+self.names[i]+"_"+name : val for name, val in results.items()})
                model.get_metrics(reset=True)