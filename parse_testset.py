#
# Copyright (c) 2020 Saarland University.
#
# This file is part of AM Parser
# (see https://github.com/coli-saar/am-parser/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Dict, Any, List, Tuple
import logging
import json

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import prepare_environment

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.common import Params

from graph_dependency_parser.components.dataset_readers.amconll_tools import parse_amconll
from graph_dependency_parser.components.dataset_readers.same_formalism_iterator import SameFormalismIterator
from graph_dependency_parser.components.evaluation.predictors import AMconllPredictor, Evaluator, StandardEvaluator
from graph_dependency_parser.graph_dependency_parser import GraphDependencyParser

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',level=logging.INFO) #turn on logging.

import graph_dependency_parser.graph_dependency_parser
import graph_dependency_parser.important_imports
import argparse

parser = argparse.ArgumentParser(description="Run the am-parser on a specified amconll file in order to annotate it, doesn't perform evaluation.")

parser.add_argument('archive_file', type=str, help='path to an archived trained model')

parser.add_argument("--batch_size", type=int, default=None, help="Overwrite batch size.")

# NOT SURE IF THESE WORK:
parser.add_argument('-k',
                       type=int,
                       default=6,
                       help='number of supertags to be used')
parser.add_argument('-t',"--threads",
                       type=int,
                       default=4,
                       help='number of threads')

parser.add_argument('--give_up',
                       type=float,
                       default=60*60,
                       help='number of seconds until fixed-tree decoder backs off to k-1')
parser.add_argument('-v',
                       action='store_true',
                       default=False,
                       help='verbose logging')

cuda_device = parser.add_mutually_exclusive_group(required=False)
cuda_device.add_argument('--cuda-device',
                         type=int,
                         default=-1,
                         help='id of GPU to use (if any)')

parser.add_argument('--weights-file',
                       type=str,
                       help='a path that overrides which weights file to use')

parser.add_argument('-o', '--overrides',
                       type=str,
                       default="",
                       help='a JSON structure used to override the experiment configuration')


args = parser.parse_args()
if args.v:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)  # turn on logging.
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

# Load from archive
archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
config = archive.config
test_evaluators : List[List[Tuple[str, Params]]] = config["test_evaluators"]
prepare_environment(config)
model = archive.model
model.eval()
if not isinstance(model, GraphDependencyParser):
    raise ConfigurationError("The loaded model seems not to be an am-parser (GraphDependencyParser)")
model : GraphDependencyParser = model

# Load the evaluation data

# Try to use the validation dataset reader if there is one - otherwise fall back
# to the default dataset_reader used for both training and validation.
validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
if validation_dataset_reader_params is not None:
    dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
else:
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))

metrics = dict()
for x in test_evaluators:
    for param_evaluator in x:
        prefix = param_evaluator[0]
        param_evaluator[1].pop("type")
        evaluator = StandardEvaluator.from_params(param_evaluator[1])

        if args.batch_size is not None:
            data_iterator = SameFormalismIterator(list(model.tasks.keys()), args.batch_size)
            evaluator.predictor.data_iterator = data_iterator

        evaluator.predictor.set_model(model)
        filename = args.archive_file + "/test_" + prefix + ".amconll"
        local_metrics = evaluator.predictor.parse_and_eval(evaluator.formalism, evaluator.system_input, evaluator.gold_file, filename=filename)
        metrics.update({prefix+"_"+k: v for k,v in local_metrics.items()})
        #evaluator.predictor.parse_and_save(evaluator.formalism, evaluator.system_input, filename)
        #evaluator.
        cumulated_parse_time = 0.0
        with open(filename) as f:
            for am_sentence in parse_amconll(f, validate=False):
                cumulated_parse_time += float(am_sentence.attributes["normalized_nn_time"]) + float(am_sentence.attributes["parsing_time"]) \
                                        + float(am_sentence.attributes["normalized_prepare_ftd_time"])
        metrics[prefix+"_time"] = cumulated_parse_time

with open(args.archive_file+"/test_metrics.json", "w") as f:
    f.write(json.dumps(metrics))

print(metrics)

logger.info("Finished parsing.")
