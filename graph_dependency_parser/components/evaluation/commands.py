from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Tuple

from allennlp.common import Params, Registrable
import subprocess
import re


class BaseEvaluationCommand(ABC, Registrable):
    """
    An evaluation command takes two files (gold and system output) and returns a dictionary with scores.
    """
    @abstractmethod
    def evaluate(self, gold_file: str, system_output:str) -> Dict[str,float]:
        raise NotImplementedError()


@BaseEvaluationCommand.register("bash_evaluation_command")
class BashEvaluationCommand(BaseEvaluationCommand):
    """
    An evaluation command that can be configured with jsonnet files.
    Executes a bash command, taps into the output and returns metrics extracted using regular expressions.
    """
    def __init__(self, command : str, result_regexes: Dict[str, str]) -> None:
        """
        Sets up an evaluator.
        :param command: a bash command that will get executed. Use {system_output} and {gold_file} as placeholders.
        :param result_regexes: a dictionary mapping metric names to regexes how to extract the values of the respective metrics.
            evaluate will return a dictionary where the keys are the metric names and the regexes are used to extract
            the respective values of the metrics. From each regex, we take the group "value". That is, use (?P<value>...) in your regex!
        """
        self.command = command
        self.result_regex = result_regexes
        for regex in result_regexes.values():
            assert "(?P<value>" in regex,f"Regex {regex} doesn't seem to contain the group ?P<value>"

    def evaluate(self, system_output: str, gold_file: str,) -> Dict[str, float]:
        """
        Calls a bash command and extracts metrics.
        :param system_output:
        :param gold_file:
        :return: a dictionary that maps metric names to their values
        """
        with subprocess.Popen([self.command.format(system_output=system_output, gold_file=gold_file)], shell=True, stdout=subprocess.PIPE) as proc:
            result = bytes.decode(proc.stdout.read())  # output of shell commmand as string
            metrics = dict()
            for metric_name, regex in self.result_regex.items():
                m = re.search(regex, result)
                if m:
                    val = float(m.group("value"))
                    metrics[metric_name] = val
            return metrics