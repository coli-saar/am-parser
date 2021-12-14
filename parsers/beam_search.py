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
    from parsers.dataset_readers.same_formalism_iterator import SameFormalismIterator
    from parsers.dataset_readers.amconll_tools import parse_amconll

    optparser = argparse.ArgumentParser(add_help=True,
                                        description="Parse an amconll file (no annotions) with beam search.")

    optparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    optparser.add_argument('input_file', type=str, help='path to or url of the input file')
    optparser.add_argument('output_file', type=str, help='path to output file')
    optparser.add_argument('--cuda-device', type=int, default=0, help='id of GPU to use. Use -1 to compute on CPU.')
    optparser.add_argument('--beam', type=int, default=2, help='beam size. Default: 2')
    optparser.add_argument("--batch_size", type=int, default=None, help="Overwrite batch size.")
    optparser.add_argument("--parse_on_cpu", action="store_true", default=False, help="Enforce parsing on the CPU.")

    args = optparser.parse_args()

    if args.beam < 1:
        print("Beam size must be at least 1")
        sys.exit()

    archive = load_archive(args.archive_file, args.cuda_device)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()
    model.k_best = args.beam
    model.parse_on_gpu = not args.parse_on_cpu

    pipelinepieces = PipelineTrainerPieces.from_params(config)

    if args.batch_size is not None and args.batch_size > 0:
        assert isinstance(pipelinepieces.annotator.data_iterator, SameFormalismIterator)
        iterator : SameFormalismIterator = pipelinepieces.annotator.data_iterator
        pipelinepieces.annotator.data_iterator = SameFormalismIterator(iterator.formalisms, args.batch_size)

    annotator = pipelinepieces.annotator
    annotator.dataset_reader.workers = 1

    #Don't read in entire AM dependency trees, just the tokens.
    annotator.dataset_reader.read_tokens_only = True

    t0 = time.time()
    annotator.annotate_file(model, args.input_file, args.output_file)
    t1 = time.time()

    cumulated_parse_time = 0.0
    with open(args.output_file) as f:
        for am_sentence in parse_amconll(f):
            cumulated_parse_time += float(am_sentence.attributes["normalized_parsing_time"])

    print("Prediction took", t1-t0, "seconds overall")
    print("Parsing time was", cumulated_parse_time)

