import torch
from typing import List, Dict, Iterable

from allennlp.data import DataIterator
from allennlp.data.dataset import Instance
import allennlp.nn.util as util

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

