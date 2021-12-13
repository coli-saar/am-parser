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
from typing import List, Optional

from allennlp.common import Registrable
from allennlp.data import Vocabulary
import torch
import abc
from torch.utils.dlpack import from_dlpack
from allennlp.nn.util import get_device_of

import spacy
from spacy.tokens import Doc
from graph_dependency_parser.components.spacy_interface import make_doc

try:
    from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer, PyTT_TokenVectorEncoder
    import cupy

    spacy.require_gpu()
except ModuleNotFoundError:
    print("Either spacy pytorch transformers or cupy not available, so you cannot use spacy-tok2vec! This is only an issue, if you intend to use roberta or xlnet.")

NAME_TO_DIM = {"xlnet-large-cased" : 1024, "xlnet-base-cased" : 768,
               "bert-base-uncased" : 768,
               "bert-large-uncased" : 1024,
               "en_pytt_xlnetbasecased_lg" : 768,
               "en_pytt_bertbaseuncased_lg" : 768,
                "en_pytt_robertabase_lg" : 768,
               "roberta-large" : 1024}

class TokenToVec(Registrable):

    @abc.abstractmethod
    def get_output_dim(self) -> int:
        """
        Returns output dimension.
        :return:
        """
        pass

    @abc.abstractmethod
    def embed(self, vocab: Vocabulary, tokens: torch.Tensor) -> torch.Tensor:
        """
        Takes a vocabulary and tensor of tokens and returns an embedded tensor
        :param vocab: the vocabulary
        :param tokens: shape (batch_size, seq len) with single ids to be interpreted in the vocabulary
        :return: a tensor of shape (batch_size, seq_len, output_dim)
        """
        pass


class SwitchDefaultTensor:
    def __enter__(self):
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_default_tensor_type("torch.FloatTensor")

@TokenToVec.register("unbatched-spacy-tok2vec")
class SpacyTokenToVec(TokenToVec):

    def __init__(self, model_name: str, model_path : Optional[str]) -> None:
        super().__init__()
        with SwitchDefaultTensor():
            if model_path:
                self.nlp = PyTT_Language(pytt_name=model_name, meta={"lang": "en"})
                self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))
                self.nlp.add_pipe(PyTT_WordPiecer.from_pretrained(self.nlp.vocab, model_name))
                self.nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(self.nlp.vocab, model_path))

            else:
                self.nlp = spacy.load(model_name)
        if not model_name in NAME_TO_DIM:
            raise ValueError("Model name is unknown, I know "+str(list(NAME_TO_DIM.keys())))
        self.output_dim = NAME_TO_DIM[model_name]

    def get_output_dim(self) -> int:
        return self.output_dim

    def embed(self, vocab: Vocabulary, tokens: torch.Tensor) -> torch.Tensor:
        """
        Idea: reconstruct string tokens from token ids -> feed to spacy -> return tensors
        :param vocab:
        :param tokens:
        :return:
        """
        with SwitchDefaultTensor():
            embedded_sentences = []
            tokens_cpu = tokens.cpu()
            batch_size, seq_len = tokens.shape
            for sentence in tokens_cpu:
                str_tokens: List[str] = [vocab.get_token_from_index(int(token)) for token in sentence if token != 0] #skip padding
                doc = Doc(self.nlp.vocab, words=str_tokens)
                self.nlp.pipeline[1][1](doc) #word pieces
                self.nlp.pipeline[2][1](doc) #run transformer on wordpieces
                #add padding back in
                #embedded = torch.from_numpy(cupy.asnumpy(doc.tensor)).to(device) # shape (str_tokens, output dim)
                embedded = from_dlpack(doc.tensor.toDlpack()) # shape (str_tokens, output dim)
                assert embedded.shape == (len(str_tokens), self.get_output_dim())
                if seq_len - len(str_tokens) > 0:
                    padded = torch.zeros(seq_len - len(str_tokens), self.get_output_dim())
                    embedded = torch.cat([embedded, padded], dim= 0)
                embedded_sentences.append(embedded)
            return torch.stack(embedded_sentences, dim=0)








@TokenToVec.register("spacy-tok2vec")
class BatchedSpacyTokenToVec(TokenToVec):

    def __init__(self, model_name: str, model_path : Optional[str] = None) -> None:
        """
        Loads a model_name (e.g. en_pytt_xlnetbasecased_lg) or a combination
        of model name and local model path (e.g. xlnet-large-cased and /local/mlinde/xlnet-large-cased)
        see https://github.com/explosion/spacy-pytorch-transformers#loading-models-from-a-path for how to prepare a model
        :param model_name:
        :param model_path:
        """
        super().__init__()
        with SwitchDefaultTensor():
            if model_path:
                self.nlp = PyTT_Language(pytt_name=model_name, meta={"lang": "en"})
                self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))
                self.nlp.add_pipe(PyTT_WordPiecer.from_pretrained(self.nlp.vocab, model_name))
                self.nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(self.nlp.vocab, model_path))

            else:
                self.nlp = spacy.load(model_name)
        if model_name not in NAME_TO_DIM:
            raise ValueError("Model name is unknown, I know "+str(list(NAME_TO_DIM.keys())))
        self.output_dim = NAME_TO_DIM[model_name]

    def get_output_dim(self) -> int:
        return self.output_dim

    def embed(self, vocab: Vocabulary, tokens: torch.Tensor) -> torch.Tensor:
        """
        Idea: reconstruct string tokens from token ids -> feed to spacy -> return tensors
        :param vocab:
        :param tokens:
        :return:
        """
        with SwitchDefaultTensor():
            with torch.autograd.no_grad():
                embedded_sentences = []
                tokens_cpu = tokens.cpu()
                batch_size, seq_len = tokens.shape
                sents = []
                for sentence in tokens_cpu:
                    str_tokens: List[str] = [vocab.get_token_from_index(int(token)) for token in sentence if token != 0] #skip padding
                    sents.append(str_tokens)
                doc = make_doc(self.nlp.vocab, sents)
                self.nlp.pipeline[1][1](doc) #word pieces
                self.nlp.pipeline[2][1](doc) #run transformer on wordpieces

                #Now iterate over sentences in correct order and cut out the correct tensor + pad it
                for sent, str_tokens in zip(doc.sents, sents):
                    #add padding back in
                    embedded = from_dlpack(sent.tensor.toDlpack()) # shape (str_tokens, output dim)
                    if seq_len - len(str_tokens) > 0:
                        padded = torch.zeros(seq_len - len(str_tokens), self.get_output_dim())
                        embedded = torch.cat([embedded, padded], dim=0)
                    embedded_sentences.append(embedded)
                return torch.stack(embedded_sentences, dim=0)