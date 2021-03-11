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
from typing import Dict, Tuple, List, Any, Iterable, TextIO, Optional
import logging
import random

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, Instance
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from graph_dependency_parser.components.dataset_readers.amconll_tools import parse_amconll, AMSentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_id2amsent_from_file(filestream: TextIO) -> Dict[str, AMSentence]:
    """
    Get a sentence ID to AMSentence mapping from an opened filestream

    :param filestream: one already opened AMCoNLL file
    :return: dictionary mapping sentence IDs(str) to AMSentence objects
    """
    id2amsentence = dict()
    for amsentence in parse_amconll(filestream):
        sentid = amsentence.attributes.get("id", None)
        if sentid is not None:  # skip if no ID
            id2amsentence[sentid] = amsentence
    return id2amsentence


def get_paired_amsentences(file1stream: TextIO, file2stream: TextIO) -> Dict[str,Tuple[AMSentence, AMSentence]]:
    """
    Get intersection of AMSentences (based on matching IDs) from two files

    :param file1stream: opened AMCoNLL file number 1 (first part of pair)
    :param file2stream: opened AMCoNLL file number 2 (second part of pair)
    :return: Dictionary mapping sentence IDs(str) to AMSentence pairs
    """
    # todo: if this function is generic enough, maybe move to another script
    # note: maybe see also analyzers/compare_amconll.py
    file1_id2sent = get_id2amsent_from_file(file1stream)
    file2_id2sent = get_id2amsent_from_file(file2stream)
    matching_ids = file1_id2sent.keys() & file2_id2sent.keys()
    id2amsentpair = dict()
    for sentid in matching_ids:
        id2amsentpair[sentid] = (file1_id2sent[sentid], file2_id2sent[sentid])
    return id2amsentpair


@DatasetReader.register("typeamconllreader")
class TypeAMConllReader(DatasetReader):
    """
    Reading pair of amconll files containing AM types: for type tagging

    Note: Sentence pairs are constructed based on matching id attribute.
    Note: AMSentences are in separate files (one per formalism), so we need
    two files as input and find matching sentences.
    For now we employ a dirty hack by assuming a certain `splitmarker` in the
    input path and use `source_target_foldername_pair` values as replacements
    to get two files from one file_path param of the `_read` function.
    """
    # We want to read from files that look something like this:
    # train:
    # superdir/PAS/train/train.amconll  AND  superdir/DM/train/train.amconll
    # dev:
    # superdir/PAS/gold-dev/gold-dev.amconll  AND  superdir/DM/gold-dev/gold-dev.amconll
    # So what we tell allenNLP (for the train case, analogous for dev):
    # train_data_path: 'superdir/' + splitmarker + '/train/train.amconll'
    # source_target_foldername_pair: ['PAS', 'DM']

    # todo: do I even need the target token indexer?
    def __init__(self, source_token_indexers: Dict[str, TokenIndexer] = None,
                 #target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_target_foldername_pair: Tuple[str, str] = ("PAS", "DM"),
                 splitmarker: str = "@@SPLITMARKER@@",
                 maxtrainsize: Optional[int] = None,
                 lazy: bool = False
                 ) -> None:
        super().__init__(lazy)
        # unlike nla_semparse, amconll tokenized, so don't need to do it :)
        self._source_token_indexers = source_token_indexers
        #self._source_token_indexers = (source_token_indexers
        #                               or {"tokens": SingleIdTokenIndexer()})
        #self._target_token_indexers = (target_token_indexers
        #                               or self._source_token_indexers)
        self.src_foldername = source_target_foldername_pair[0]
        self.tgt_foldername = source_target_foldername_pair[1]
        self.splitmarker = splitmarker
        self.maxtrainsize = maxtrainsize
        self.use_small_train = self.maxtrainsize is not None
        if self.use_small_train:
            assert(self.maxtrainsize > 0)
            logger.info("Max trainsize set to %s ", self.maxtrainsize)
        return

    # todo: what happens for test with gold output?
    def _read_src_tgt_filepair(self, src_file: str, tgt_file: str) -> Iterable[Instance]:
        # src_file and tgt_file are both amconll files
        # todo: what is not copied from amconll.py: chached_path
        # todo: what is not copied from amconll.py: only_read_fraction_of_train
        id2amsentpair = dict()
        with open(src_file, 'r') as src_f, open(tgt_file, 'r') as tgt_f:
            logger.info("Reading AM types from amconll dataset at: %s (source) and %s (target)", src_file, tgt_file)
            # note: overlap for the two files is computed based on matching ids
            id2amsentpair = get_paired_amsentences(src_f, tgt_f)
        # collect all sentence IDs and eventually reduce the training set:
        used_sent_ids = id2amsentpair.keys()
        if self.use_small_train and src_file.endswith("train.amconll") and self.maxtrainsize < len(used_sent_ids):  # todo: only when train? what about dev?
            logger.info("Only pick %s (randomly sampled) out of %s total train sentences", self.maxtrainsize, len(used_sent_ids))
            used_sent_ids = random.sample(used_sent_ids, k=self.maxtrainsize)
        # convert sentence pairs to instances
        for sentid in used_sent_ids:
            yield self.text_to_instance(id2amsentpair[sentid])

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # how to join filepath common prefix with individual suffixes?
        # dirty hack: file_path contains SPLITMARKER (hopefully): split path
        # there to insert foldernames for the two formalisms so we can have
        # two files (one per formalism) with just one file_path input param
        # todo: find a better solution for the two-files-as-input problem
        if not file_path.count(self.splitmarker) == 1:
            raise RuntimeError("Filepath should contain the `splitmarker` "
                               "specified at initialization time exactly once!")
        prefix, suffix = file_path.split(self.splitmarker)
        # plain string concat below todo do I need pathlib or the like?
        src_file = prefix + self.src_foldername + suffix
        tgt_file = prefix + self.tgt_foldername + suffix
        for instance in self._read_src_tgt_filepair(src_file, tgt_file):
            yield instance
    
    @overrides
    def text_to_instance(self, src_tgt_amsentpair: Tuple[AMSentence, AMSentence]) -> Instance:
        # todo: what if tgt types not there??? (test/ignored in dev?)
        # todo: should there be a start,end token? no types for them?
        src_amsent, tgt_amsent = src_tgt_amsentpair
        assert(len(src_amsent) == len(tgt_amsent))
        fields: Dict[str, Field] = {}

        # note: a SequenceLabelField contains labels for the elements in the
        # corresponding TextField
        # todo: token indexers?
        src_tokens = TextField(tokens=[Token(w) for w in src_amsent.get_tokens(shadow_art_root=True)], token_indexers=self._source_token_indexers)
        # tgt_tokens = TextField([Token(w) for w in tgt_amsent.get_tokens(shadow_art_root=True)], self._target_token_indexers)

        fields["src_words"] = src_tokens
        # fields["tgt_words"] = tgt_tokens
        fields["src_pos"] = SequenceLabelField(src_amsent.get_pos(), src_tokens, label_namespace="src_pos")
        # optional: ner, lemmas, (supertags?, lexlabels? head_tags?...)
        fields["src_types"] = SequenceLabelField(src_amsent.get_types(), src_tokens, label_namespace="src_types")

        # todo: assert target_tokens are the same as src tokens?
        fields["tgt_types"] = SequenceLabelField(tgt_amsent.get_types(), src_tokens, label_namespace="tgt_types")

        fields["metadata"] = MetadataField({
            "src_words": src_amsent.words, "tgt_words": tgt_amsent.words,  # AMSentence.words is a List[Entry]
            "src_attributes": src_amsent.attributes,
            "tgt_attributes": tgt_amsent.attributes})
        # "formalism": formalism, "position_in_corpus" : position_in_corpus,
        # "token_ranges" : am_sentence.get_ranges(),
        # "is_annotated" : am_sentence.is_annotated()})

        # from nla_semparse  (included start, end token for output)
        # tokenized_source = self._source_tokenizer.tokenize(source_string)
        # source_field = TextField(tokenized_source, self._source_token_indexers)
        # if target_string is not None:
        #     tokenized_target = self._target_tokenizer.tokenize(target_string)
        #     tokenized_target.insert(0, Token(START_SYMBOL))
        #     tokenized_target.append(Token(END_SYMBOL))
        #     target_field = TextField(tokenized_target,
        #                              self._target_token_indexers)
        #     return Instance(
        #         {"source_tokens": source_field, "target_tokens": target_field})
        # else:
        #     return Instance({"source_tokens": source_field})
        return Instance(fields)


def main():
    print("(Debugging) Testing the dataset reader...")
    source_token_indexers = {
        "tokens": SingleIdTokenIndexer(namespace="source_tokens")
    }
    # target_token_indexers = {
    #     "tokens": SingleIdTokenIndexer(namespace="target_tokens")
    # }
    splitmarker = "@@SPLITMARKER@@"
    dataset_reader = TypeAMConllReader(
        source_token_indexers=source_token_indexers,
        # target_token_indexers=target_token_indexers,
        source_target_foldername_pair= ("PAS", "DM"),
        splitmarker=splitmarker
    )
    path = "./toydata/" + splitmarker + "/gold-dev/gold-dev.amconll"
    # path = "./toydata/" + SPLITMARKER + "/train/train.amconll"
    #path = "./toydata/train/toy_train_"
    # todo: which read function?
    instances = dataset_reader._read(path)

    for instance in instances:
        print(instance)
        # field = instance["src_pos"]
        # field = instance.get("src_words")
        # print(field[0].text)
        # print(field[0]==field[3])  # pos: NNP == NNP
        # print(field[0]==field[1])  # pos: NNP !=
    print("--done--")


if __name__ == "__main__":
    main()
