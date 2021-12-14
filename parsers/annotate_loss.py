import argparse
from typing import List, Dict, Any

import torch
from allennlp.common.util import prepare_environment, import_submodules
from allennlp.data import Instance
from allennlp.models import load_archive
from allenpipeline import Annotator, OrderedDatasetReader, PipelineTrainerPieces
from allenpipeline.Decoder import split_up
import allennlp.nn.util as util


if __name__ == "__main__":
    import_submodules("topdown_parser")
    from topdown_parser.nn.parser import TopDownDependencyParser

    optparser = argparse.ArgumentParser(add_help=True,
                                        description="Annotates loss into an annotated amconll file. Was used to determine if beam search would help.")

    optparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    optparser.add_argument('input_file', type=str, help='path to or url of the input file')
    optparser.add_argument('output_file', type=str, help='path to output file')
    optparser.add_argument('--cuda-device', type=int, default=0, help='id of GPU to use. Use -1 to compute on CPU.')


    class LossAnnotator(Annotator):
        """
        Computes the loss of the instances
        """
        def annotate(self, model : TopDownDependencyParser, instances : List[Instance]) -> List[Dict[str, Any]]:
             with torch.no_grad():
                 self.data_iterator.index_with(model.vocab)
                 cuda_device = model._get_prediction_device()
                 preds = []
                 for dataset in self.data_iterator._create_batches(instances,shuffle=False):
                     dataset.index_instances(model.vocab)
                     model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
                     output_dict = model.annotate_loss(**model_input) # INSTEAD of forward, call annotate_loss
                     output_dict = split_up(output_dict, model_input["order_metadata"])
                     preds.extend(output_dict)

                 if self.decoder:
                     preds = self.decoder.decode_batch(model.vocab, preds)

                 return OrderedDatasetReader.restore_order(preds)


    args = optparser.parse_args()

    archive = load_archive(args.archive_file, args.cuda_device)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    pipelinepieces = PipelineTrainerPieces.from_params(config)

    loss_annotator = LossAnnotator(pipelinepieces.annotator.data_iterator, pipelinepieces.annotator.dataset_reader, pipelinepieces.annotator.dataset_writer)
    loss_annotator.dataset_reader.workers = 1
    loss_annotator.annotate_file(model, args.input_file, args.output_file)

