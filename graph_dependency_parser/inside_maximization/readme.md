# Experiments

## AMR

### Baseline

This is simply rerunning the L19 code to get it into comet.ml and to double check it.

This was run on tony-1 in /local/jonasg/unsupervised2020/am-parser

Step 1: set up server as described in wiki, clone repo
Step 2: `cp -r /proj/corpora/abstract_meaning_representation_amr_2.0_LDC2017T10/abstract_meaning_representation_amr_2.0/data/amrs/split/* /local/jonasg/unsupervised2020/AMR17/corpus/`
Step 3: in `/local/jonasg/unsupervised2020/am-parser/`: `bash scripts/preprocess-no-baseline.sh -m /local/jonasg/unsupervised2020/AMR17/ &>/local/jonasg/unsupervised2020/AMR17/screenLog.txt`
