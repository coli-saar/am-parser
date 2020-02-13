import os
from tempfile import TemporaryDirectory
from typing import Dict, Any
import logging
import json

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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',level=logging.INFO) #turn on logging.

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
parser.add_argument('-t',"--threads",
                       type=int,
                       default=1,
                       help='number of threads')

parser.add_argument("--port",
                    type=int,
                    default=8888,
                    help='number of threads')

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

parser.add_argument('--embedding-sources-mapping',
                       type=str,
                       default="",
                       help='a JSON dict defining mapping from embedding module path to embedding'
                       'pretrained-file used during training. If not passed, and embedding needs to be '
                       'extended, we will try to use the original file paths used during training. If '
                       'they are not available we will use random vectors for embedding extension.')

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


predictor = AMconllPredictor(dataset_reader,args.k,args.give_up, args.threads, model=model)

requires_art_root = {"DM" : True, "PAS": True, "PSD": True, "EDS" : False, "AMR-2015": False, "AMR-2017": False}
requires_ne_merging = {"DM" : False, "PAS": False, "PSD": False, "EDS" : False, "AMR-2015": True, "AMR-2017": True}


import asyncio
import json

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
    o_fil = ""
    if formalism == "DM" or formalism == "PAS":
        os.system(f"java -cp {args.am_tools} de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus -c {filename} -o {filename}_o")
        o_fil = f"{filename}_o.sdp"

    elif formalism == "PSD":
        os.system(f"java -cp {args.am_tools} de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPCorpus -c {filename} -o {filename}_o")
        o_fil = f"{filename}_o.sdp"

    elif formalism == "EDS":
        os.system(f"java -cp {args.am_tools} de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus -c {filename} -o {filename}_o")
        o_fil = f"{filename}_o.amr.txt"

    elif "AMR" in formalism:
        os.system(f"bash scripts/eval_AMR_new.sh {filename} {output_path} {args.am_tools}")
        # python /local/mlinde/mtool/main.py --normalize edges --read amr --write dot parserOut.txt o.dot
        #creates output_path/parserOut.txt
        o_fil = f"{output_path}/parserOut.txt"
    else:
        return f"ERROR: formalism {formalism} not known."

    with open(o_fil) as f:
        text = f.read()

    return text



async def handle_client(reader, writer):
    request = (await reader.read(255)).decode('utf8') #read 255 characters
    print("Request",request)
    json_req = json.loads(request)
    print("-- as json",json_req)
    sentence = json_req["sentence"]
    formalisms = json_req["formats"]
    words = spacy_tokenize(sentence)

    with TemporaryDirectory() as direc:
        ret_val = {"sentence" : sentence, "parses" : { f : {} for f in formalisms}}

        for formalism in formalisms:
            if formalism not in model.tasks:
                print("Model was not trained on",formalism,"but on", model.tasks)
                continue

            #Create input and save to file:
            sentences = [from_raw_text(sentence.rstrip("\n"),words,requires_art_root[formalism], dict(),requires_ne_merging[formalism])]
            temp_path = direc+f"/sentences_{formalism}.amconll"
            output_filename = direc+"/parsed_"+formalism+".amconll"

            with open(temp_path,"w") as f:
                for s in sentences:
                    f.write(str(s))
                    f.write("\n\n")

            predictor.parse_and_save(formalism, temp_path, output_filename)

            #Read AM dependency tree
            with open(output_filename) as f:
                ret_val["parses"][formalism]["amdep"] = f.read()

            #Evaluate to graph
            raw_graph = postprocess(output_filename, direc, formalism)
            ret_val["parses"][formalism]["graph"] = raw_graph

    writer.write(bytes(json.dumps(ret_val),"utf8"))
    await writer.drain()
    writer.close()

loop = asyncio.get_event_loop()
loop.create_task(asyncio.start_server(handle_client, 'localhost', args.port))
loop.run_forever()
