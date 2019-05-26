from collections import deque
from typing import Iterable, Deque, List, Tuple
import logging
import random
from collections import defaultdict

from allennlp.common.util import lazy_groups_of
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.iterators import BucketIterator
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def split_by_formalism(instance_list):
    insts_by_lang = defaultdict(lambda: [])
    for inst in instance_list:
        inst_lang = inst.fields['metadata'].metadata['formalism']
        insts_by_lang[inst_lang].append(inst)

    return insts_by_lang

@DataIterator.register("same_formalism")
class SameFormalismIterator(DataIterator):
    """

    Splits batches into batches containing the same formalism and uses several underlying bucket iterators.
    On initialization, the formalisms must be provided as a list of strings.

    """
    def __init__(self, formalisms: List[str],
                 batch_size,
                 cache_instances:bool = False,
                 track_epoch:bool = False,
                 padding_noise:float = 0.1,
                 instances_per_epoch:int = None,
                 max_instances_in_memory: int  = None,
                 maximum_samples_per_batch: Tuple[str, int] = None,
                 biggest_batch_first: bool = False):

        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self.bucket_iterators = dict()
        for formalism in formalisms:
            self.bucket_iterators[formalism] = BucketIterator(sorting_keys=[("words","num_tokens")],
                                                              batch_size=batch_size,
                                                              track_epoch=track_epoch,
                                                              padding_noise=padding_noise,
                                                              biggest_batch_first=biggest_batch_first)

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab
        for formalism in self.bucket_iterators:
            self.bucket_iterators[formalism].index_with(vocab)

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            available_formalisms = list(self.bucket_iterators.keys())
            formalism_specific_iterator = [None] * len(available_formalisms)
            instances_by_formalism = split_by_formalism(instance_list)
            for formalism, instances in instances_by_formalism.items():
                formalism_specific_iterator[available_formalisms.index(formalism)] = self.bucket_iterators[formalism]._create_batches(instances, shuffle)

            available_batches = np.array([self.bucket_iterators[f].get_num_batches(instances_by_formalism[f]) for f in available_formalisms])
            while any(x > 0 for x in available_batches):
                #select formalism of batch with probability proportional to number of batches left of this formalism.
                formalism = np.random.choice(range(len(available_formalisms)), p = available_batches/np.sum(available_batches))
                batch = next(formalism_specific_iterator[formalism])
                available_batches[formalism] -= 1
                yield batch

