# Experiments

## AMR

### Baseline

This is simply rerunning the L19 code to get it into comet.ml and to double check it.

This was run on tony-1 in /local/jonasg/unsupervised2020/am-parser

Step 1: set up server as described in wiki, clone repo (used commit 3aec3990191452b4734a4dd32ff4ab30111c3b3b in unsupervised2020 branch)

Step 2: copy a current version of the `jar` file from the am-tools `unsupervised2020` branch to the repo (I used commit `68d8142738716ffcb96342be9f0df9e618f1a5b7`).

Step 3: `cp -r /proj/corpora/abstract_meaning_representation_amr_2.0_LDC2017T10/abstract_meaning_representation_amr_2.0/data/amrs/split/* /local/jonasg/unsupervised2020/AMR17/corpus/`

Step 4: in `/local/jonasg/unsupervised2020/am-parser/`: `bash scripts/preprocess-no-baseline.sh -m /local/jonasg/unsupervised2020/AMR17/ &>/local/jonasg/unsupervised2020/AMR17/screenLog.txt`

Step 5: set all AMR-17 related datapaths in `configs/data_paths.libsonnet`, `configs/eval_commands.libsonnet`, `configs/test_evaluators.libsonnet` and `configs/validation_evaluators.libsonnet` to the ones in `/local/jonasg/unsupervised2020/AMR17/`.

Step 6: run ` python -u train.py jsonnets/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/saved_models/amr17baseline/  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  5  } }' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --project unsupervised2020-amr  &> /local/jonasg/unsupervised2020/saved_models/amr17baseline/screenLog.txt`
