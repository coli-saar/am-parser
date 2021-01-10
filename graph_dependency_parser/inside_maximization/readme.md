# Experiments

## AMR

### Baseline

This is simply rerunning the L19 code to get it into comet.ml and to double check it.

This was run on tony-1 in /local/jonasg/unsupervised2020/am-parser

Step 1: set up server as described in wiki, clone repo (used commit 3aec3990191452b4734a4dd32ff4ab30111c3b3b in unsupervised2020 branch)

Step 2: copy a current version of the `jar` file from the am-tools `unsupervised2020` branch to the repo (I used commit `68d8142738716ffcb96342be9f0df9e618f1a5b7`).

Step 3: Copy original AMR corpus via `cp -r /proj/corpora/abstract_meaning_representation_amr_2.0_LDC2017T10/abstract_meaning_representation_amr_2.0/data/amrs/split/* /local/jonasg/unsupervised2020/AMR17/corpus/`

Step 4: Run AMR preprocessing: in `/local/jonasg/unsupervised2020/am-parser/` run `bash scripts/preprocess-no-baseline.sh -m /local/jonasg/unsupervised2020/AMR17/ &>/local/jonasg/unsupervised2020/AMR17/screenLog.txt`

Step 5: set all AMR-17 related datapaths in `configs/data_paths.libsonnet`, `configs/eval_commands.libsonnet`, `configs/test_evaluators.libsonnet` and `configs/validation_evaluators.libsonnet` to the ones in `/local/jonasg/unsupervised2020/AMR17/`.

Step 6: run training via ` python -u train.py jsonnets/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/saved_models/amr17baseline/  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  5  } }' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --project unsupervised2020-amr  &> /local/jonasg/unsupervised2020/saved_models/amr17baseline.log`

Note: this seems to have stopped for no reason I can see. Rerunning with ` python -u train.py jsonnets/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/saved_models/amr17baseline2/  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  6  } }' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --project unsupervised2020-amr  &> /local/jonasg/unsupervised2020/saved_models/amr17baseline2.log` (also changed eval threads to 8 for dev set, up from 4)


### Automata with given alignments, L19 pre/post

Running the automata creation on falken-3:

Step 1: Copy a current version of the `jar` file from the am-tools `unsupervised2020` branch to `/local/jonasg/unsupervised2020/` (I used commit `68d8142738716ffcb96342be9f0df9e618f1a5b7` as in the baseline above).

Step 2: Copy results of Step 4 from above baseline to falken-3: `scp -r /local/jonasg/unsupervised2020/AMR17 falken-3:/local/jonasg/unsupervised2020/` (note: didn't end up in quite the right place due to typo, had to fiddle with it; the command here should be right); same for stanford models that were downloaded in baseline experiment: `scp -r /local/jonasg/unsupervised2020/am-parser/stanford falken-3:/local/jonasg/unsupervised2020/`

Step 3: Create the automata: in `/local/jonasg/unsupervised2020/` run `java -Xmx800G -cp am-tools.jar de.saar.coli.amtools.decomposition.SourceAutomataCLIAMR -t AMR17/data/alto/train/namesDatesNumbers_AlsFixed_sorted.corpus -d AMR17/data/alto/dev/namesDatesNumbers_AlsFixed_sorted.corpus -o AMR17/automata/ --stanford-ner-model stanford/english.conll.4class.distsim.crf.ser.gz --stanford-pos-model stanford/english-bidirectional-distsim.tagger -s 4 -a automata &>AMR17/automata.log`

__this hasn't run yet, will need to update commit and commands. In fact, this experiment may not be necessary__

### creating toy dataset

Run on my own computer, using WSL and the allennlp coda env. Working directory is `Work`.

`java -Xmx4G -cp GitHub/am-tools/build/libs/am-tools.jar de.saar.coli.amrtagging.formalisms.amr.tools.DependencyExtractorCLI -c experimentData/unsupervised2020/AMR17/toyAMR.corpus -li 10 -o experimentData/unsupervised2020/AMR17/toyAMR/nn/ -t 1 -pos experimentData/unsupervised2020/stanford/english-bidirectional-distsim.tagger 2>&1 | tee experimentData/unsupervised2020/AMR17/toyAMR/dep.log`

`java -Xmx4G -cp GitHub/am-tools/build/libs/am-tools.jar de.saar.coli.amrtagging.formalisms.amr.tools.ToAMConll -c experimentData/unsupervised2020/AMR17/toyAMR/nn/ -o experimentData/unsupervised2020/AMR17/toyAMR/ --stanford-ner-model experimentData/unsupervised2020/stanford/english.conll.4class.distsim.crf.ser.gz --no-lexlabel-replacement 2>&1 | tee experimentData/unsupervised2020/AMR17/toyAMR/amconll.log`

