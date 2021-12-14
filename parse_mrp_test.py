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

from parsers.components.evaluation.predictors import AMconllPredictor, Evaluator
from parsers.graph_dependency_parser import GraphDependencyParser

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',level=logging.INFO) #turn on logging.

import parsers.graph_dependency_parser
import parsers.important_imports
import argparse

parser = argparse.ArgumentParser(description="Run the am-parser on its amconll test file in order to annotate it, doesn't perform evaluation. Used to parse test set in MRP shared task.")

parser.add_argument('archive_file', type=str, help='path to an archived trained model')
parser.add_argument('config_file', type=str, help='config file with test evaluator')

parser.add_argument('-k',
                       type=int,
                       default=6,
                       help='number of supertags to be used')
parser.add_argument('-t',"--threads",
                       type=int,
                       default=4,
                       help='number of threads')

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

parser.add_argument('--embedding-sources-mapping',
                       type=str,
                       default="",
                       help='a JSON dict defining mapping from embedding module path to embedding'
                       'pretrained-file used during training. If not passed, and embedding needs to be '
                       'extended, we will try to use the original file paths used during training. If '
                       'they are not available we will use random vectors for embedding extension.')

args = parser.parse_args()
# Disable some of the more verbose logging statements
logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

# Load from archive
archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
archive_config = archive.config
config = Params.from_file(args.config_file,args.overrides)
prepare_environment(config)
model = archive.model
model.eval()
if not isinstance(model, GraphDependencyParser):
    raise ConfigurationError("The loaded model seems not to be an am-parser (GraphDependencyParser)")

# Load the evaluation data

# Try to use the validation dataset reader if there is one - otherwise fall back
# to the default dataset_reader used for both training and validation.
validation_dataset_reader_params = archive_config.pop('validation_dataset_reader', None)
if validation_dataset_reader_params is not None:
    dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
else:
    dataset_reader = DatasetReader.from_params(archive_config.pop('dataset_reader'))

embedding_sources: Dict[str, str] = (json.loads(args.embedding_sources_mapping)
                                     if args.embedding_sources_mapping else {})

test_evaluators = config.pop("test_evaluators",[])
test_evaluators : List[Tuple[str, Evaluator]] = [(name, Evaluator.from_params(evaluator)) for formalism in test_evaluators for name, evaluator in formalism]
if len(test_evaluators) == 0:
    logger.warning("No test evaluators were given, after training, there will be no evaluation on a test set.")

logger.info("The model will be evaluated using the best epoch weights.")


for _,evaluator in test_evaluators:
    evaluator.eval(model,float("inf"), args.archive_file)

logger.info("Finished parsing.")
