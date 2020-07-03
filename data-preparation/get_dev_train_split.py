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

# todo do conversion mrp to amr inside this script? not done yet (output currently mrp)
# todo does this work on a server?
# todo: make random seed, filename-id separator and rounding function cmd line param?

from math import ceil, floor
import sys         # for exit, stderr, 
import os          # os.path.isdir, os.path.exists, mkdir ...
import argparse    # parse command line options
import random      # choose graphs for dev randomly


# todo: make variables below command line parameter?
separator = "___"   # separates filename and original id
# roundingfunction = ceil
roundingfunction = floor
randomseed = 1
random.seed(randomseed)
verbose = True
# verbose = False


# for printing to stderr
def eprint(*args, **kwargs):
    """prints to standard error stream (use it as normal print)"""
    print(*args, file=sys.stderr, **kwargs)


def get_all_corpus_files_from_dir(dirname: str, prefix="", suffix="") -> list:
    """
    Get a filtered list of all files in dirname directory

    :param dirname: directory where to search for corpus files
    :param prefix  string, filter consider only files with a certain prefix
    :param suffix  string, filter consider only files with a certain suffix
    :returns all corpus file paths in the given directory
    """
    files = [os.path.join(dirname, f) for f in os.listdir(dirname)
             if os.path.isfile(os.path.join(dirname, f))
             and f.startswith(prefix) and f.endswith(suffix)]
    return sorted(files)  # todo: sorted to have reproducible order - necess?


def main():
    # 1. get cmd arguments
    defaultdevpercentage = 3.0  # 3 percent
    defaulttrainfilename = "traincorpus.amr"
    defaultdevfilename = "devcorpus.amr"
    optparser = argparse.ArgumentParser(add_help=True, 
        description="split corpus directory in dev and train")
    optparser.add_argument("inputdir", type=str,
                           help="Path to input directory")
    optparser.add_argument("devdir", type=str,
                           help="Path to dev (output) directory")
    optparser.add_argument("traindir", type=str,
                           help="Path to train (output) directory")
    optparser.add_argument("--devpercentage", dest="devpercentage", type=float,
                           default=defaultdevpercentage,
                           help="Dev percentage (default: {} meaning {} percent)".format(
                               defaultdevpercentage, defaultdevpercentage))
    # optparser.print_help()
    opts = optparser.parse_args()
    inputdir = opts.inputdir
    devdir = opts.devdir
    traindir = opts.traindir
    percentage = opts.devpercentage
    
    if verbose:
      print("->   INPUT DIRECTORY: ", inputdir)
      print("->     DEV DIRECTORY: ", devdir)
      print("->   TRAIN DIRECTORY: ", traindir)
      print("->    Dev percentage: ", percentage)
      print("-> Rounding function: ", roundingfunction)
      print("->       Random seed: ", randomseed)
    
    # 2. Check whether input meets requirements
    for dir in [devdir, traindir]: # create train dev dir if necessary
      if not os.path.exists(dir):
        os.mkdir(dir)
    for dir in [inputdir, devdir, traindir]:  # stop if dir doesn't exist
      if not os.path.isdir(dir):
          eprint("Not a valid directory path: ", dir)
          sys.exit(2)
    # check percentage between 0 and 100, excluding the boundaries 0 or 100 % dev doesn't make sense
    if percentage <= 0 or percentage >= 100:
      eprint("Percentage needs to be in (0,100)! Type 5 for 5 percent. Exit")
      sys.exit(2)
    assert(percentage != 0)
    percentage /= 100  # convert to percentage   from 3 to 0.03
    
    # Get all mrp files in indir
    mrpfiles = get_all_corpus_files_from_dir(dirname=inputdir, prefix="", suffix=".mrp")
    if len(mrpfiles) == 0:
      eprint("No corpus files found - cannot generate dev/train corpus!")
      sys.exit(2)
    if verbose:
      print("-> Corpus files found: ", len(mrpfiles))
    # print("Corpus files found: ", mrpfiles)  # debug
    
    # get dev and train file path
    devfile = os.path.join(devdir, defaultdevfilename)
    trainfile = os.path.join(traindir, defaulttrainfilename)
    # for file in [devfile, trainfile]: # clear train/dev files
      # # todo: chatch permission denied errors?
      # f = open(file, "w")
      # f.close()
    
    devcnt = 0    # number of graphs in dev set
    traincnt = 0  # number of graphs in train set
    with open(devfile, mode="w", encoding="utf-8") as devout, \
        open(trainfile, mode="w", encoding="utf-8") as trainout:
      for mrpfile in mrpfiles:  # for each input file
        filename = os.path.basename(mrpfile)  # without path to it
        fname = filename.replace("\"","_")
        # todo: catch permission denied errors?
        graphcnt = 0
        # 1. first run through the corpus : get number of graphs
        with open(mrpfile, mode="r", encoding="utf-8") as infile:
          linenumber = 0
          for line in infile:
            linenumber +=1 
            if not line.startswith("{\"id\": \"") and not line.strip() == "":
              eprint("File {} : Line {}: unexpected line format {}".format(mrpfile, linenumber, line))
              continue
            graphcnt += 1
        
        
        # 2. select graphs to be assigned to dev set
        if graphcnt == 0:
          eprint("No graphs found in file ", mrpfile, " - ignore and continue")
          continue
        assert(graphcnt > 0)
        
        # 2.1. calculate the number of graphs for this file that should be assigned to dev
        number_of_devgraphs = roundingfunction(percentage * graphcnt)
        if number_of_devgraphs == 0:
          eprint("For file {} and percentage {} , 0 graphs to dev - write at least one to dev".format(mrpfile, percentage))
          number_of_devgraphs = 1
        # print("Unrounded ", percentage * graphcnt)  # debug
        # print("Rounded ", number_of_devgraphs)
        # print("I.e. x percent where x = ", number_of_devgraphs / graphcnt)
        assert(number_of_devgraphs <= graphcnt)
        if verbose:
          print(": {:>7} dev graphs (out of {:>7} total) for file {:>30} :".format(number_of_devgraphs, graphcnt, mrpfile))
        
        # 2.2 select concrete graphs for the dev set:
        # assume that graphs in a file are numbered from 1 to graphcnt, so let's pick 
        graphnumbers_for_dev = set(random.sample(range(1, graphcnt+1), number_of_devgraphs))
        assert(len(graphnumbers_for_dev) == number_of_devgraphs)
        devcnt += number_of_devgraphs
        traincnt += (graphcnt - number_of_devgraphs)
        
        # 3. second run through the corpus : change graph id and print to dev/train
        with open(mrpfile, mode="r", encoding="utf-8") as infile:
          linenumber = 0
          graphno = 0
          for line in infile:
            linenumber +=1 
            if line.rstrip() == "":  # skip empty line
              continue
            # {"id": "DF-201-185522-35_2114.33", "flavor": 2,
            lenstart = len("{\"id\": \"")
            if not line.startswith("{\"id\": \""):
              eprint("File {} : Line {}: unexpected line format {}".format(mrpfile, linenumber, line))
              continue
            assert(len(line) > lenstart+3)  # {"id": "x"}
            graphno += 1
            # add filename to original id 
            #newl = line[:lenstart] + filename + separator + line[lenstart:]
            newl = line
            # write new graph dev or train file
            if graphno in graphnumbers_for_dev:
              devout.write(newl)  # todo: do I need to add "\n" ?
            else:
              trainout.write(newl)
        # mrpf file opened
      # for each mrp file
    # dev and train open
    print("Total number of graphs: {:>10}".format(devcnt+traincnt))
    print("dev   number of graphs: {:>10}".format(devcnt))
    print("train number of graphs: {:>10}".format(traincnt))
    print("dev percentage    (actual): {:<10}".format(devcnt / (devcnt + traincnt)))
    print("dev percentage (requested): {:<10}".format(percentage))
    print("Remember rounding function used: ", roundingfunction)
    
    print("\n --> next step in amr pipeline: mrp 2 amr conversion <--")
    return


if __name__ == "__main__":
    main()
