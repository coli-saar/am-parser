import os
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import List, Iterable, Dict, Tuple

from allennlp.common import Params, Registrable
import subprocess
import re
import json

from graph_dependency_parser.components.utils import flatten, merge_dicts


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
    def __init__(self, command : str, result_regexes: Dict[str, str], show_output: bool = True) -> None:
        """
        Sets up an evaluator.
        :param command: a bash command that will get executed. Use {system_output} and {gold_file} as placeholders.
        :param result_regexes: a dictionary mapping metric names to tuples of line number and regexes how to extract the values of the respective metrics.
            evaluate will return a dictionary where the keys are the metric names and the regexes are used to extract
            the respective values of the metrics in the specified lines. From each regex, we take the group "value". That is, use (?P<value>...) in your regex!
        :param if output of evaluation command should be printed.
        """
        self.command = command
        self.result_regex = result_regexes
        self.show_output = show_output
        for line_number,regex in result_regexes.values():
            assert "(?P<value>" in regex,f"Regex {regex} doesn't seem to contain the group ?P<value>"

    def evaluate(self, system_output: str, gold_file: str) -> Dict[str, float]:
        """
        Calls a bash command and extracts metrics.
        :param system_output:
        :param gold_file:
        :return: a dictionary that maps metric names to their values
        """
        with TemporaryDirectory() as direc:
            cmd = self.command.format(system_output=system_output, gold_file=gold_file, tmp=direc)
            with subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE) as proc:
                result = bytes.decode(proc.stdout.read())  # output of shell commmand as string
                result_lines = result.split("\n")
                if self.show_output:
                    print(result)
                metrics = dict()
                for metric_name, (line_number,regex) in self.result_regex.items():
                    m = re.search(regex, result_lines[line_number])
                    if m:
                        val = float(m.group("value"))
                        metrics[metric_name] = val
                if self.show_output:
                    print(metrics)
                return metrics




@BaseEvaluationCommand.register("amr_evaluation_command")
class AMREvaluationCommand(BaseEvaluationCommand):
    """
    An evaluation command for AMR that can be configured with jsonnet files.
    """
    def __init__(self, amr_year : str, tool_dir : str, alto_path: str, show_output: bool = True) -> None:
        """
        Sets up an evaluator.
        :param amr_year: 2015 or 2017
        :param tool_dir: the path to the evaluation tools used for AMR (2019rerun)
        :param alto_path: the path to the Alto .jar file
        :param show_output: show Smatch results on commmand line?
        """
        self.amr_year = amr_year
        assert amr_year in ["2015","2017"]
        self.tool_dir = tool_dir
        self.alto_path = alto_path
        self.show_output = show_output

    def evaluate(self, system_output: str, gold_file: str) -> Dict[str, float]:
        """
        Calls the evaluation functions and returns extracted metrics.
        :param system_output:
        :param gold_file:
        :return: a dictionary that maps metric names to their values
        """
        assert gold_file in ["dev","test"], f"In case of AMR, set gold_file in the validation_evaluator to dev or test (got {gold_file})"
        with TemporaryDirectory() as direc:
            os.system(f"java -cp {self.alto_path} de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus -c {system_output} -o {direc}")
            if "dev" == gold_file:
                if self.amr_year == "2017":
                    os.system(f"bash {self.tool_dir}/scripts/eval_dev17.sh {direc} {self.alto_path}")
                else:
                    os.system(f"bash {self.tool_dir}/scripts/eval_dev.sh {direc} {self.alto_path}")
            elif "test" == gold_file:
                if self.amr_year == "2017":
                    os.system(f"bash {self.tool_dir}/scripts/eval_test17.sh {direc} {self.alto_path}")
                else:
                    os.system(f"bash {self.tool_dir}/scripts/eval_test.sh {direc} {self.alto_path}")
            else:
                raise ValueError(f"Given gold file {gold_file} I can't determine if this is dev or test data")
            metrics = dict()
            with open(direc + "/smatch.txt") as f:
                lines = f.readlines()
                for line in lines:
                    name, score = line.split(": ")
                    metrics[name] = 100 * float(score)
            if self.show_output:
                print (metrics)
            return metrics


@BaseEvaluationCommand.register("json_evaluation_command")
class JsonEvaluationCommand(BaseEvaluationCommand):
    """
    An evaluation command that can be configured with jsonnet files.
    Executes a bash command, taps into the output and returns metrics extracted using json.
    """
    def __init__(self, commands : List[List[str]], show_output: bool = True) -> None:
        """
        Sets up an evaluator.
        :param commands: a list of pairs of (metric_prefix, command) that will get executed. Use {system_output} and {gold_file} and {tmp} as placeholders.
        {tmp} points to a private temporary directory. if metric_prefix is the empty string, no metric will be saved.
        :param if output of evaluation command should be printed.
        """
        self.commands = commands
        for cmd in self.commands:
            assert len(cmd) == 2, "Should get a tuple of [metric_prefix, command] but got "+str(cmd)
        self.show_output = show_output

    def evaluate(self, system_output: str, gold_file: str) -> Dict[str, float]:
        """
        Calls the bash commands and extracts metrics for
        :param system_output:
        :param gold_file:
        :return: a dictionary that maps metric names to their values
        """
        metrics = dict()
        with TemporaryDirectory() as direc:
            for prefix,cmd in self.commands:
                cmd = cmd.format(system_output=system_output, gold_file=gold_file, tmp=direc)
                with subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE) as proc:
                    result = bytes.decode(proc.stdout.read())  # output of shell commmand as string
                    if self.show_output:
                        print(result)
                    if prefix:
                        try:
                            result_json = json.loads(result)
                            metrics = merge_dicts(metrics, prefix, flatten(result_json))
                        except json.decoder.JSONDecodeError: #probably not intended for us
                            if self.show_output:
                                print("<-- not well-formed json, ignoring")

        if self.show_output:
            print(metrics)
        return metrics