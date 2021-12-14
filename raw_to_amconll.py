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
from parsers.components.dataset_readers.amconll_tools import from_raw_text
from parsers.components.spacy_interface import spacy_tokenize

import argparse

parser = argparse.ArgumentParser(description="""Take raw text and generate an empty amconll file that can then be parsed. 
                                               NOTE: this is not the method we used to prepare the test set, so results may differ.
                                               If you want to parse the test set (or see what kind of pre-processing is done there),
                                               have a look at scripts/predict.sh
                                               """)

parser.add_argument('text_file', type=str, help='Path to raw text file. We assume one sentence per line.')
parser.add_argument("output_file",type=str, help="Path to amconll file that will be written.")
parser.add_argument("formalism",type=str,help="Which formalism to prepare the amconll for. For instance, DM, PAS, PSD need an additional 'artificial token' at the end of each sentence. Available are: DM, PAS, PSD, EDS, AMR.")

parser.add_argument('--no-tok',action="store_true",
                    help="Don't use spacy tokenizer and split on spaces.")

args = parser.parse_args()


requires_art_root = {"DM": True, "PAS": True, "PSD": True, "EDS": False, "AMR": False}
requires_ne_merging = {"DM": False, "PAS": False, "PSD": False, "EDS": False, "AMR": True}

if args.formalism not in requires_art_root:
    raise ValueError(f"Do not recognize the formalism {args.formalism}; known are: {sorted(requires_art_root.keys())}")

sents = []
with open(args.text_file) as f:
    for sentence in f:
        if args.no_tok:
            sentence = sentence.strip()
            words = sentence.split(" ")
        else:
            words = spacy_tokenize(sentence)
        am_sentence = from_raw_text(sentence.rstrip("\n"), words, requires_art_root[args.formalism], dict(),
                                    requires_ne_merging[args.formalism])
        sents.append(am_sentence)

with open(args.output_file, "w") as f:
    for s in sents:
        f.write(str(s))
        f.write("\n\n")



