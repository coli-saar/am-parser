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
from allennlp.training.util import evaluate

from parsers.components.evaluation.predictors import AMconllPredictor, Evaluator
from parsers.graph_dependency_parser import GraphDependencyParser

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',level=logging.INFO) #turn on logging.

import parsers.graph_dependency_parser
import parsers.important_imports
import argparse

parser = argparse.ArgumentParser(description="Run the am-parser on a specified amconll file in order to compute LAS, UAS and supertagging accuracies.")

parser.add_argument('formalism', type=str, help='name of formalism (must be included in the model)')
parser.add_argument('input_file', type=str, help='path to input file with gold annotations')
parser.add_argument('archive_file', type=str, help='path to an archived trained model')

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

parser.add_argument('--batch-weight-key',
                       type=str,
                       default="",
                       help='If non-empty, name of metric used to weight the loss on a per-batch basis.')

args = parser.parse_args()
# Disable some of the more verbose logging statements
logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

# Load from archive
archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
config = archive.config
prepare_environment(config)
model = archive.model
model.eval()
if not isinstance(model, GraphDependencyParser):
    raise ConfigurationError("The loaded model seems not to be an am-parser (GraphDependencyParser)")

# Load the evaluation data

# Try to use the validation dataset reader if there is one - otherwise fall back
# to the default dataset_reader used for both training and validation.
validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
if validation_dataset_reader_params is not None:
    dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
else:
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))

instances = dataset_reader.read([[args.formalism, args.input_file]])

iterator_params = config.pop("validation_iterator", None)
if iterator_params is None:
    iterator_params = config.pop("iterator")
iterator = DataIterator.from_params(iterator_params)
iterator.index_with(model.vocab)

metrics = evaluate(model, instances, iterator, args.cuda_device, args.batch_weight_key)

logger.info("Finished evaluating.")
logger.info("Metrics:")
for key, metric in metrics.items():
    logger.info("%s: %s", key, metric)
