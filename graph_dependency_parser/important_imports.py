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

import jnius_config
jnius_config.add_options('-Xmx50G')
jnius_config.set_classpath('am-tools.jar')
import jnius

import graph_dependency_parser.components.dataset_readers.same_formalism_iterator
import graph_dependency_parser.components.dataset_readers.amconll
import graph_dependency_parser.components.dataset_readers.amconll_automata

import graph_dependency_parser.components.evaluation
import graph_dependency_parser.components.AMTask
import graph_dependency_parser.inside_maximization.AMAutomataTask
import graph_dependency_parser.inside_maximization.graph_dependency_parser_automata
import graph_dependency_parser.am_algebra
import graph_dependency_parser.components.losses
import graph_dependency_parser.components.edge_models
