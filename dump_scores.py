import socket
from typing import Dict, Any
import logging
import json
import re

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import prepare_environment

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.common import Params

from graph_dependency_parser.components.dataset_readers import amconll_tools
from graph_dependency_parser.components.dataset_readers.amconll_tools import AMSentence
from graph_dependency_parser.components.dataset_readers.same_formalism_iterator import SameFormalismIterator
from graph_dependency_parser.components.evaluation.iterator import forward_on_instances
from graph_dependency_parser.components.evaluation.predictors import AMconllPredictor
from graph_dependency_parser.components.supertagger import Supertagger
from graph_dependency_parser.graph_dependency_parser import GraphDependencyParser

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',level=logging.INFO) #turn on logging.

import graph_dependency_parser #important import
import argparse
import zipfile
import numpy as np
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Run the neural network to get scores supertags and edges.")

parser.add_argument('archive_file', type=str, help='path to an archived trained model')
parser.add_argument('formalism', type=str, help='name of formalism (must be included in the model)')
parser.add_argument('input_file', type=str, help='path to the amconll file containing the sentences to be processed')
parser.add_argument('output_path', type=str, help='path and name of zip file where to store the scores.')

cuda_device = parser.add_mutually_exclusive_group(required=False)
cuda_device.add_argument('--cuda-device',
                         type=int,
                         default=-1,
                         help='id of GPU to use (if any)')



cuda_device.add_argument('--supertag-limit',
                         type=int,
                         default=15,
                         help='How many labels per edge to include in the scores file')

parser.add_argument('--edge-label-limit',
                         type=int,
                         default=30,
                         help='How many labels per edge to include in the scores file')

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

parser.add_argument('--extend-vocab',
                       action='store_true',
                       default=False,
                       help='if specified, we will use the instances in your new dataset to '
                            'extend your vocabulary. If pretrained-file was used to initialize '
                            'embedding layers, you may also need to pass --embedding-sources-mapping.')

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
config = archive.config
prepare_environment(config)
model = archive.model
model.eval()
if not isinstance(model, GraphDependencyParser):
    raise ConfigurationError("The loaded model seems not to be an am-parser (GraphDependencyParser)")
if not args.formalism in model.tasks:
    raise ConfigurationError(f"The model at hand was not trained on {args.formalism} but on {list(model.tasks.keys())}")

# Load the evaluation data

# Try to use the validation dataset reader if there is one - otherwise fall back
# to the default dataset_reader used for both training and validation.
validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
if validation_dataset_reader_params is not None:
    dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
else:
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
evaluation_data_path = args.input_file

embedding_sources: Dict[str, str] = (json.loads(args.embedding_sources_mapping)
                                     if args.embedding_sources_mapping else {})
formalism = args.formalism

if args.extend_vocab:
    logger.info("Vocabulary is being extended with test instances.")
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read([[formalism,evaluation_data_path]])
    model.vocab.extend_from_instances(Params({}), instances=instances)
    model.extend_embedder_vocab(embedding_sources)

instances = dataset_reader.read([[formalism, args.input_file]])  # we need to give the formalism to amconll dataset_reader
model.train(False)
data_iterator = DataIterator.from_params(config.pop('iterator'))

with open (args.input_file) as f:
    conll_sentences = list(amconll_tools.parse_amconll(f))

predictions = dataset_reader.restore_order(forward_on_instances(model, instances, data_iterator))

i2edge_label = [model.vocab.get_token_from_index(i, namespace=formalism + "_head_tags") for i in
                range(model.vocab.get_vocab_size(formalism + "_head_tags"))]

i2supertag = [model.vocab.get_token_from_index(i, namespace=formalism+"_supertag_labels")
              for i in range(model.vocab.get_vocab_size(formalism+"_supertag_labels"))]

lexlabel2i = { model.vocab.get_token_from_index(i, namespace=formalism+"_lex_labels") : i
              for i in range(model.vocab.get_vocab_size(formalism+"_lex_labels"))}

def dump_tags(score, fragment, type):
    if type == "_": #\bot
        x = "NULL"
    else:
        x = fragment.replace("  "," ").replace(" ","__ALTO_WS__")+"--TYPE--"+str(type).replace(" ","")
    return x+"|"+str(round(score,5))

top_k_labels = args.edge_label_limit
top_k_supertags = args.supertag_limit
bot_id = model.vocab.get_token_index(AMSentence.get_bottom_supertag(),namespace=formalism+"_supertag_labels")


with zipfile.ZipFile(args.output_path,"w",compression=zipfile.ZIP_DEFLATED, compresslevel=7) as myzip:
    tagprobs = []
    modified_conll_sentences = []
    with myzip.open("opProbs.txt","w") as fp:
        for sentence_id,pred in enumerate(predictions):

            if "supertag_scores" in pred:
                all_supertag_scores = F.log_softmax(torch.from_numpy(pred["supertag_scores"]),1) #shape (sent length, num supertags)
                top_k_supertag_indices = torch.argsort(all_supertag_scores, descending=True, dim=1)[:, :top_k_supertags].numpy()
                all_supertag_scores = all_supertag_scores.numpy()

            edge_scores = np.transpose(pred["edge_existence_scores"],[1,0]) #shape (sent len+1 (from), sent len+1 (to))
            mask = 1e9*np.eye(edge_scores.shape[0])
            edge_scores = edge_scores - mask #mask out diagonal
            edge_scores = F.log_softmax(torch.from_numpy(edge_scores),dim=0).numpy() #normalize over incoming edges.

            edge_label_scores = np.transpose(pred["full_label_logits"],[1,0,2]) #shape (sent len+1, sent len+1, num edge labels). Semantics: from, to, label index
            edge_label_scores = F.log_softmax(torch.from_numpy(edge_label_scores),dim=2).numpy() #normalize over edge labels

            modified_sent = conll_sentences[sentence_id].set_heads(pred["predicted_heads"])
            if "lexlabels" in pred:
                modified_sent = modified_sent.set_lexlabels(pred["lexlabels"])

            attributes = pred["attributes"]
            attributes["batch_size"] = str(pred["batch_size"])
            attributes["normalized_nn_time"] = str(pred["batch_time"] / pred["batch_size"])
            attributes["nn_host"] = socket.gethostname()
            modified_sent.attributes = attributes

            modified_conll_sentences.append(modified_sent)

            #ja = False
            #if attributes["id"] == "#22048021":
            #    print(edge_scores.shape)
            #    print(list(edge_scores[:,4]))
            #    print(list(edge_scores[:,5]))
            #    print("---")
            #    print(edge_scores)
            #    ja = True


            sent_length = edge_scores.shape[0] #sent length + art root
            edges = []
            for from_ in range(sent_length):
                for to_ in range(sent_length):
                    if from_ == to_:
                        continue
                    o = f"[{from_},{to_}]|{edge_scores[from_,to_]:.4f} "
                    interesting_labels = sorted(enumerate(edge_label_scores[from_,to_]),key=lambda x:x[1],reverse=True)[:top_k_labels]
                    o += " ".join([i2edge_label[lbl]+f"|{score:.4f}" for lbl,score in interesting_labels])
                    edges.append(o)
            #if ja:
            #    print(edges)
            fp.write("\t".join(edges).encode())
            fp.write("\n".encode())

            if "supertag_scores" in pred:
                tokens = []
                for word in range(sent_length-1):
                    supertags_for_this_word = []
                    for top_k in top_k_supertag_indices[word]:
                        fragment, typ = AMSentence.split_supertag(
                            model.vocab.get_token_from_index(top_k, namespace=formalism + "_supertag_labels"))
                        score = all_supertag_scores[word, top_k]
                        supertags_for_this_word.append((score, fragment, typ))
                    if bot_id not in top_k_supertag_indices[word]:  # \bot is not in the top k, but we have to add it anyway in order for the decoder to work properly.
                        fragment, typ = AMSentence.split_supertag(AMSentence.get_bottom_supertag())
                        supertags_for_this_word.append((all_supertag_scores[word, bot_id], fragment, typ))
                    tokens.append(" ".join([dump_tags(score,fragment,type) for (score, fragment, type) in supertags_for_this_word]))
                tagprobs.append("\t".join(tokens)) #have to keep this in memory because we can only have one file open in zipfile at a time

    with myzip.open("tagProbs.txt", "w") as tp:
        for sent in tagprobs:
            tp.write(sent.encode())
            tp.write("\n".encode())

    with myzip.open("corpus.amconll","w") as am:
        for sent in modified_conll_sentences:
            am.write(str(sent).encode())
            am.write("\n\n".encode())




logger.info("Finished dumping scores.")
