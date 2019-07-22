"""
Takes an MRP file and spits writes the MRP graphs of the devset into a second corpus.

who to blame: matthias
"""
import argparse
import json
import random
import sys

optparser = argparse.ArgumentParser(add_help=True, 
    description="creates dev split for MRP file and spits out ids of devset sentences")
optparser.add_argument("file", help="input MRP corpus", type=str)
optparser.add_argument("file2", help="MRP dev corpus, will be created", type=str)
optparser.add_argument("file3", help="MRP train corpus, will be created", type=str)
optparser.add_argument("--fraction", type=float,default=3.0,
                       help="what PERCENTAGE of corpus should (roughly) go into dev data?")
opts = optparser.parse_args()

random.seed(11)

with open(opts.file) as f:
    with open(opts.file2,"w") as f2:
        with open(opts.file3,"w") as f3:
            for line in f:
                sentence = json.loads(line)
                sample = random.random() * 100
                if sample < opts.fraction:
                    f2.write(line)
                else:
                    f3.write(line)
