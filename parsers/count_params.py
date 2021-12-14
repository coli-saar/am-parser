import argparse
import sys
import time
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
    from topdown_parser.dataset_readers.same_formalism_iterator import SameFormalismIterator

    optparser = argparse.ArgumentParser(add_help=True,
                                        description="Count trainable parameters.")

    optparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')

    args = optparser.parse_args()


    archive = load_archive(args.archive_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model

    total_params = 0
    # for module in model.modules():
    #     params = module.parameters()
    #     print(module, sum(p.numel() for p in params if p.requires_grad))
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()

    print(round(total_params/1_000_000,2),"M", "parameters")

