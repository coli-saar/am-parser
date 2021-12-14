from typing import Optional

from allennlp.common.checks import ConfigurationError
from allennlp.models import Model
from allenpipeline import Annotator, BaseEvaluationCommand
from allenpipeline.callback import Callback
from comet_ml import Experiment


@Callback.register("parse-dev")
class ParseDev(Callback):
    """
    This callback allows us to parse the (full) dev set after every epoch.
    """
    def __init__(self, eval_command : BaseEvaluationCommand, system_input : str, prefix : Optional[str] = ""):
        self.system_input = system_input
        self.eval_command = eval_command
        self.prefix = prefix

        if self.eval_command.gold_file is None:
            raise ConfigurationError("Callback requires a gold file to be set.")

    def call(self, annotator : Annotator, model : Model, trainer : Optional["PipelineTrainer"] = None, experiment : Optional[Experiment] = None):
        filename = trainer._serialization_dir+f"/full_dev_epoch_{trainer.epoch}.txt"
        annotator.annotate_file(model, self.system_input, filename)
        results = self.eval_command.evaluate(filename)
        # Hack into trainer object and amend metrics.
        trainer.val_metrics.update({self.prefix+name : val for name, val in results.items()})
        model.get_metrics(reset=True)

@Callback.register("create-checkpoint")
class CreateCheckpoint(Callback):
    """
    This callback simply creates a checkpoint.
    """
    def call(self, annotator : Annotator, model : Model, trainer : Optional["PipelineTrainer"] = None, experiment : Optional[Experiment] = None):
        if trainer is not None:
            trainer._save_checkpoint("before-validation")