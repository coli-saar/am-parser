import argparse
import json
from typing import Dict, List

from allennlp.common.util import prepare_environment, import_submodules
from allennlp.models import load_archive
from allenpipeline import PipelineTrainerPieces
from allenpipeline.callback import CallbackName


# Example:
# python topdown_parser/extract_time.py some_file.amconll


if __name__ == "__main__":
    import_submodules("topdown_parser")
    from topdown_parser.dataset_readers.amconll_tools import parse_amconll

    optparser = argparse.ArgumentParser(add_help=True,
                                        description="Extract parsing time from parsed amconll file.")

    optparser.add_argument('input', type=str, help='amconll file with time measurements')

    args = optparser.parse_args()

    time_dict : Dict[str, List[float]] = dict()
    with open(args.input) as f:
        for am_sentence in parse_amconll(f, validate=False):
            for k, v in am_sentence.attributes.items():
                if "time" in k:
                    try:
                        f = float(v)
                        if k not in time_dict:
                            time_dict[k] = []
                        time_dict[k].append(f)
                    except ValueError:
                        pass

    total_time = 0
    print("="*80)
    for k, v in time_dict.items():
        t = sum(v)
        total_time += t
        print(k, t)
    print("="*80)
    print("total time", total_time)



