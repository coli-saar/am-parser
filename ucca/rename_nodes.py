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
from get_edges_from_mrp import get_mrp_edges,get_id2lex
import sys


def rename_nodes(edge_dict_processed, edge_dict_gold, label_dict_processed, label_dict_gold):
    to_rename = []
    unused_gold_nodes = []
    processed_us = [u for (u, v) in edge_dict_processed.keys()]
    processed_vs = [v for (u, v) in edge_dict_processed.keys()]
    for identifier in list(label_dict_processed.keys()):
        if identifier.startswith("NONTERMINAL"):
            to_rename.append(identifier)
    for identifier in list(label_dict_gold.keys()):
        if identifier not in us and idenfier not in vs:
            unused_gold_nodes.append(identifier)
    to_rename_match_scores = {}
    for node in unused_gold_nodes:
