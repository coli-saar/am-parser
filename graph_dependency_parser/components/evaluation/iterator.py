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
import torch
import copy
from typing import List, Dict, Iterable

from allennlp.data import DataIterator
from allennlp.data.dataset import Instance
import allennlp.nn.util as util
from allennlp.nn.util import get_text_field_mask


from graph_dependency_parser.components.cle import cle_decode

import numpy


def forward_on_instances(model,
                         instances: Iterable[Instance], data_iterator: DataIterator) -> List[Dict[str, numpy.ndarray]]:
    """
    Basically a copy of Model.forward_on_instances, but also takes a DataIterator in order to be more efficient.


    Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
    arrays using this model's :class:`Vocabulary`, passes those arrays through
    :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
    and returns the result.  Before returning the result, we convert any
    ``torch.Tensors`` into numpy arrays and separate the
    batched output into a list of individual dicts per instance. Note that typically
    this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
    :func:`forward_on_instance`.

    Parameters
    ----------
    model : AllenNLP model, required
        The model to run.
    instances : List[Instance], required
        The instances to run the model on.
    data_iterator: DataIterator, required
        The DataIterator used for going over the data (e.g. BucketIterator)

    Returns
    -------
    A list of the models output for each instance.
    """
    data_iterator.index_with(model.vocab)
    with torch.no_grad():
        return_val: List[Dict[str, numpy.ndarray]] = []
        cuda_device = model._get_prediction_device()
        for dataset in data_iterator._create_batches(instances, shuffle=False):
            batch_size = len(dataset.instances)
            dataset.index_instances(model.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = model.decode(model(**model_input))
            instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in dataset.instances]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        model._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    model._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return_val.extend(instance_separated_output)
        return return_val



def forward_on_instances_interpolation(model, pretrained_model, beta, formalism,
                         instances: Iterable[Instance], data_iterator: DataIterator) -> List[Dict[str, numpy.ndarray]]:
    """
    Basically a copy of Model.forward_on_instances, but also takes a DataIterator in order to be more efficient.


    Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
    arrays using this model's :class:`Vocabulary`, passes those arrays through
    :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
    and returns the result.  Before returning the result, we convert any
    ``torch.Tensors`` into numpy arrays and separate the
    batched output into a list of individual dicts per instance. Note that typically
    this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
    :func:`forward_on_instance`.

    Parameters
    ----------
    model : AllenNLP model, required
        The model to run.
    instances : List[Instance], required
        The instances to run the model on.
    data_iterator: DataIterator, required
        The DataIterator used for going over the data (e.g. BucketIterator)

    Returns
    -------
    A list of the models output for each instance.
    """
    data_iterator.index_with(model.vocab)
    # data_iterator.index_with(pretrained_model.vocab)
    with torch.no_grad():
        return_val: List[Dict[str, numpy.ndarray]] = []
        cuda_device = model._get_prediction_device()
        for dataset in data_iterator._create_batches(instances, shuffle=False):
            batch_size = len(dataset.instances)
            dataset_bk = copy.deepcopy(dataset)

            dataset.index_instances(model.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            # print(model_input['lemmas'])
            output_dict_1 = model(**model_input)

            # for instance in dataset.instances:
            #     for field in instance.fields.values():
            #         field.index(pretrained_model.vocab)
            dataset_bk.index_instances(pretrained_model.vocab)
            model_input2 = util.move_to_device(dataset_bk.as_tensor_dict(), cuda_device)
            # print(model_input2['lemmas'])
            # raise NotImplementedError()
            output_dict_2 = pretrained_model(**model_input2)

            interpolation(output_dict_1, output_dict_2, beta, model_input, model, pretrained_model, formalism)

            outputs = model.decode(output_dict_1)
            instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in dataset.instances]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        model._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    model._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return_val.extend(instance_separated_output)
        return return_val

def interpolation(output_dict_1, output_dict_2, beta, dataset, model1, model2, formalism):
    edge_existence_scores1 = output_dict_1['edge_existence_scores']
    edge_existence_scores2 = output_dict_2['edge_existence_scores'] 
    edge_existence_scores = edge_existence_scores1 * (1-beta) + edge_existence_scores2 * beta
    words = dataset['words']
    mask = get_text_field_mask(words)
    predicted_heads = cle_decode(edge_existence_scores, mask.data.sum(dim=1).long())
    edge_label_logits1 = model1.tasks[formalism].edge_model.label_scores(output_dict_1['encoded_text_parsing'], predicted_heads)
    edge_label_logits2 = model2.tasks[formalism].edge_model.label_scores(output_dict_2['encoded_text_parsing'], predicted_heads)
    edge_label_logits = edge_label_logits1 * (1-beta) + edge_label_logits2 * beta
    output_dict_1['edge_existence_scores'] = edge_existence_scores
    output_dict_1['edge_label_logits'] = edge_label_logits
