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
import os
import sys
import time
import traceback
from io import StringIO
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Dict, Any
import logging
import json

import jnius_config

import torch

import asyncio

torch.set_num_threads(1)

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import prepare_environment

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.common import Params

from graph_dependency_parser.components.dataset_readers.amconll_tools import from_raw_text, parse_amconll
from graph_dependency_parser.components.evaluation.predictors import AMconllPredictor
from graph_dependency_parser.components.spacy_interface import spacy_tokenize
from graph_dependency_parser.graph_dependency_parser import GraphDependencyParser

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)  # turn on logging.

import graph_dependency_parser.graph_dependency_parser
import graph_dependency_parser.important_imports
import argparse

parser = argparse.ArgumentParser(description="Run the am-parser as a server.")

parser.add_argument('archive_file', type=str, help='path to an archived trained model')

parser.add_argument('am_tools', type=str, help='path to am-tools.jar')

parser.add_argument('-k',
                    type=int,
                    default=6,
                    help='number of supertags to be used')
parser.add_argument('-t', "--threads",
                    type=int,
                    default=1,
                    help='number of threads')

parser.add_argument("--port",
                    type=int,
                    default=8888,
                    help='Port to be used')

parser.add_argument("--lookup",
                    type=str,
                    default="downloaded_models/lookup/lookupdata17/",
                    help='Path to AMR-2017 lookup data.')

parser.add_argument("--wordnet",
                    type=str,
                    default="downloaded_models/wordnet3.0/dict/",
                    help='Path to wordnet')

parser.add_argument('--give_up',
                    type=float,
                    default=1,
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
config.formalism = "DUMMY"
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

predictor = AMconllPredictor(dataset_reader, args.k, args.give_up, args.threads, model=model)

print("Server is ready.")

requires_art_root = {"DM": True, "PAS": True, "PSD": True, "EDS": False, "AMR-2015": False, "AMR-2017": False}
requires_ne_merging = {"DM": False, "PAS": False, "PSD": False, "EDS": False, "AMR-2015": True, "AMR-2017": True}

jnius_config.set_classpath(".", args.am_tools)
from jnius import autoclass


class AMToolsInterface:
    def evaluate(self, input_file: str, output_path: str) -> str:
        raise NotImplementedError()


class DMInterface(AMToolsInterface):
    def __init__(self):
        self.main = autoclass("de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPJsonl")

    def evaluate(self, input_file: str, output_path: str) -> str:
        save_to = input_file + "_o"
        self.main.main(["-c", input_file, "-o", save_to])
        return save_to + ".jsonl"


class PSDInterface(AMToolsInterface):
    def __init__(self):
        self.main = autoclass("de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPJsonl")

    def evaluate(self, input_file: str, output_path: str) -> str:
        save_to = input_file + "_o"
        self.main.main(["-c", input_file, "-o", save_to])
        return save_to + ".jsonl"


class EDSInterface(AMToolsInterface):
    def __init__(self):
        self.main = autoclass("de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus")

    def evaluate(self, input_file: str, output_path: str) -> str:
        save_to = input_file + "_o"
        self.main.main(["-c", input_file, "-o", save_to])
        return save_to + ".amr.txt"


class AMRInterface(AMToolsInterface):
    def __init__(self, lookupdata: str, wordnet_path: str):
        self.lookupdata = lookupdata
        self.wordnet_path = wordnet_path
        self.main = autoclass("de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus")

    def evaluate(self, input_file: str, output_path: str) -> str:
        self.main.main(
            ["-c", input_file, "-o", output_path, "--relabel", "--wn", self.wordnet_path, "--lookup", self.lookupdata,
             "--th", "10"])
        return output_path + "/parserOut.txt"


formalism_to_class = {"DM": DMInterface(), "PAS": DMInterface(), "PSD": PSDInterface(), "EDS": EDSInterface(),
                      "AMR-2017": AMRInterface(args.lookup, args.wordnet)}


def postprocess(filename, output_path, formalism):
    """
    if [ "$type" = "DM" ] || [ "$type" = "PAS" ]; then
        java -cp $jar de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus -c $amconll -o $output$type
    elif [ "$type" = "PSD" ]; then
        java -cp $jar de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPCorpus -c $amconll -o $output$type
    elif [ "$type" = "EDS" ]; then
        java -cp $jar de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus -c $amconll -o "$output"$type
    elif [ "$type" = "AMR-2017" ]; then
        bash scripts/eval_AMR_new.sh $amconll $output $jar
    fi
    """
    t = time.time()
    o_fil = formalism_to_class[formalism].evaluate(filename, output_path)
    format = ""
    if formalism in {"DM", "PSD", "PAS"}:
        format = "dm"
    elif formalism == "EDS":
        format = "amr"
    elif "AMR" in formalism:
        format = "amr"
    else:
        return f"ERROR: formalism {formalism} not known.", ""

    with open(o_fil) as f:
        text = f.read()

    if formalism in {"DM", "PSD", "PAS"}:
        text = json.loads(text) # convert json string into an actual json object

    graph_time = time.time() - t

    t = time.time()
    # Create svg file.
    return (text, graph_time)


async def handle_client(reader, writer):
    request = (await reader.read(4048)).decode('utf8')  # read a maximum of 4048 bytes, that's more than enough
    print("Request", request)
    ret_val = {"errors": []}
    # times: amdep: parse time, svg: time to visualize graph, graph: evaluation time from amdep to graph, amdep-svg: viz. of amdep tree.
    t1 = time.time()
    try:
        json_req = json.loads(request)
        print("-- as json", json_req)
        sentence = json_req["sentence"]
        if len(sentence) > 256:
            raise ValueError("Your input exceeded the maximal input length")

        formalisms = json_req["formats"]
        words = spacy_tokenize(sentence)

        with TemporaryDirectory() as direc:
            ret_val["sentence"] = sentence
            ret_val["parses"] = {f: {} for f in formalisms}

            for formalism in formalisms:
                if formalism not in model.tasks:
                    err = f"Model was not trained on '{formalism}' but on {list(model.tasks.keys())}"
                    print(err)
                    ret_val["errors"].append(err)
                    continue

                if formalism not in requires_art_root:
                    err = f"Server doesn't know how to handle '{formalism}' although the model was trained on it."
                    print(err)
                    ret_val["errors"].append(err)
                    continue

                t = time.time()
                # Create input and save to file:
                sentences = [from_raw_text(sentence.rstrip("\n"), words, requires_art_root[formalism], dict(),
                                           requires_ne_merging[formalism])]
                temp_path = direc + f"/sentences_{formalism}.amconll"
                output_filename = direc + "/parsed_" + formalism + ".amconll"

                with open(temp_path, "w") as f:
                    for s in sentences:
                        f.write(str(s))
                        f.write("\n\n")

                predictor.parse_and_save(formalism, temp_path, output_filename)

                # Read AM dependency tree
                with open(output_filename) as f:
                    ret_val["parses"][formalism]["amdep"] = f.read()
                ret_val["parses"][formalism]["parse_time"] = time.time() - t

                # Evaluate to graph
                raw_graph, graph_time = postprocess(output_filename, direc, formalism)
                graph_format = "penman" if formalism in {"AMR-2017", "EDS"} else "sdp"
                ret_val["parses"][formalism][graph_format] = raw_graph
                ret_val["parses"][formalism]["postprocess_time"] = graph_time

    except BaseException as ex:  #
        err = "".join(traceback.TracebackException.from_exception(ex).format_exception_only())
        ret_val["errors"].append(err)
        print("Ignoring error:")
        print(err)

    writer.write(bytes(json.dumps(ret_val), "utf8"))
    await writer.drain()
    writer.close()
    t2 = time.time()
    print("Handling request took", t2 - t1)


loop = asyncio.get_event_loop()
loop.create_task(asyncio.start_server(handle_client, 'localhost', args.port))
loop.run_forever()
