#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""
Stripping all 'primitives' from the input corpus in TSV format

Background:
  The COGS dataset (specifically the train files) contain some samples that
  consist of only one word ('primitives'). In the 3rd column of these samples,
  you'll find the string `primitive` (vs `in_distribution`/`exposure example_*`)
  For `train.tsv` that's 143 samples (out of >24k samples),
  for `train_100.tsv` that's 14300 samples (out of 39.5k samples)

Rationale:
  right now (May 2021) some primitives pose a challenge to our preprocessing aka
  transformation to graphs (specifically verbs due to open sources).
  We strip all primitives from the training data to get first results.
  THIS IS ONLY MEANT AS A TEMPORARY HACK!!!!
  To solve COGS, we would need to incorporate the primitive samples, because
  5 of the 21 generalization types of COGS deal with primitives.

author:
  pia

usage:
  python3 stripprimitives.py --cogsdatadir COGSDATADIR

  will create new train_noprim.tsv and train_100_noprim.tsv files in COGSDATADIR
"""

import sys  # for argc,argv and exit
import os   # file and directory paths
import csv  # read TSV files
import argparse


def strip_primitives_from_file(inputfile, outputfile):
    primitives_found = 0
    print(f"--Stripping primitives from input file {inputfile} "
          f"and writing to {outputfile}")
    with open(inputfile, "r", encoding="utf-8") as infile, \
            open(outputfile, "w", encoding="utf-8") as outfile:
        infilereader = csv.reader(infile, delimiter='\t')
        for row in infilereader:
            if not len(row) == 3:  # sentence, logical form, gen-type
                print("Row doesn't have length 3 (ignore): ", row)
                continue  # todo print some warning?
            if row[2] == "primitive":
                primitives_found += 1
                pass
            else:
                # todo use csv module for outputfile too instead of join() ?
                outfile.write("\t".join(row)+"\n")
    print(f"  #Primitives stripped: {primitives_found}")
    return


def main(argv):
    output_postfix = "_noprim"   # --> train_noprim.tsv
    files_with_prim = ["train", "train_100"]

    parser = argparse.ArgumentParser(description='Strip primitives from COGS')
    parser.add_argument('--cogsdatadir', type=str,
                        help="COGS data directory, e.g. '../COGS/data/'")

    args = parser.parse_args(argv)
    inputdir = args.cogsdatadir

    # input validation
    if not os.path.isdir(inputdir):
        raise NotADirectoryError(f"Input path is not a directory: {inputdir}")
    for file in files_with_prim:
        full_path = os.path.join(inputdir, file+".tsv")
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

    # strip primitives from the respective files
    for file in files_with_prim:
        # create new file in same directory as input file
        input_file = os.path.join(inputdir, file+".tsv")
        output_file = os.path.join(inputdir, file+output_postfix+".tsv")
        strip_primitives_from_file(inputfile=input_file, outputfile=output_file)
    print("Done!")
    return


if __name__ == "__main__":
    main(sys.argv[1:])  # strip scriptname
