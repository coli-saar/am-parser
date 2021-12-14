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

from parsers.components.dataset_readers.amconll_tools import parse_amconll
from parsers.components.dataset_readers.same_formalism_iterator import SameFormalismIterator
from parsers.components.evaluation.predictors import AMconllPredictor, Evaluator, StandardEvaluator
from parsers.graph_dependency_parser import GraphDependencyParser

import parsers.graph_dependency_parser
import parsers.important_imports
import argparse

parser = argparse.ArgumentParser(description="Count parameters in trained model.")

parser.add_argument('archive_file', type=str, help='path to an archived trained model')

args = parser.parse_args()


archive = load_archive(args.archive_file)
config = archive.config
prepare_environment(config)
model = archive.model

total_params = 0
# for module in model.modules():
#     params = module.parameters()
#     print(module, sum(p.numel() for p in params if p.requires_grad))
for p in model.parameters():
    if p.requires_grad:
        total_params += p.numel()

print(round(total_params/1_000_000,2),"M", "parameters")
