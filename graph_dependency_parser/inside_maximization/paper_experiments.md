# Experiments in the paper: Reproduction on the Saarland server

## Basic setup and Overview

Use the conda environment `allennlp` as described here: https://github.com/coli-saar/am-parser/wiki/Setup-and-file-locations-on-the-Saarland-servers#first-time-setup (i.e. this corresponds to the basic setup of the `am-parser` repo as described in the main readme)

There are two steps to each experiment: (1) decomposition, and (2) training (and evaluating) the neural parser. For decomposition, we use `am-tools`, for parsing `am-parser`. The decomposition was run on the falken-servers, and the parsing on jones (1-3) and tony-1. The parsing experiments are tracked on comet.ml at https://www.comet.ml/jgroschwitz/unsupervised2020 (for SDP) and at https://www.comet.ml/jgroschwitz/unsupervised2020-amr (for AMR). All experiments were run from `/proj/irtg/sempardata/unsupervised2020/` with either the `am-tools.jar` in that directory or the `am-parser` folder in that directory. Results of the decomposition are in `/proj/irtg/sempardata/unsupervised2020/amconll/` and results of training the neural parser are in `/local/jonasg/unsupervised2020/temp/` of the corresponding server.

**Important note:** A lot of the bash scripts used here have input and output paths in `/proj/irtg/sempardata/unsupervised2020/` hardcoded in them. If you want to reproduce them, I strongly suggest you replace those paths, so that you don't overwrite the original output (I guess the original output would just be replaced with the same thing, but if something goes wrong it would be nice to still have the originals). This applies to all scripts in `/proj/irtg/sempardata/unsupervised2020/scripts/` and in `/proj/irtg/sempardata/unsupervised2020/am-parser/scripts/unsupervised2020/`.

## Decomposition

To reproduce the decomposition runs, compile the `new_decomposition` branch of `am-tools` as of commit `5d43339`, or use the `am-tools.jar` in `/proj/irtg/sempardata/unsupervised2020/` (note: there is also an `am-tools.jar` in `/proj/irtg/sempardata/unsupervised2020/am-parser/`, but that is from the master branch to enable easier compatibility with the neural parser. I'll sort this out when I merge the two branches -- in fact, I think this was only because of the PSD preprocessing legacy option, which I already included in the `new_decomposition` branch more recently).

### Neural joint learning

To create the tree automata files from the input files (for SDP from the raw files, for AMR more complicated, see below), have a look at `/proj/irtg/sempardata/unsupervised2020/scripts/createDMAutomata.sh`, which looks like this:

```
nrSources=$1
amtools="am-tools.jar"
outpath="amconll/Auto$nrSources"

. scripts/storeFilePaths.sh
mkdir $outpath
mkdir $outpath/DM
# DM
dmCommand="java -Xmx800G -cp $amtools de.saar.coli.amtools.decomposition.SourceAutomataCLI -t $DM_TRAIN -d $DM_DEV -s $nrSources -o $outpath/DM/ -a automata -ct DM &>$outpath/DM/log.txt"
echo "processing DM with the following command"
echo $dmCommand

eval $dmCommand
```

It first does some setup, describing output path and the path to the jar file, and loading input filepaths on the server from `/proj/irtg/sempardata/unsupervised2020/scripts/storeFilePaths.sh`. Then it prints and evaluates the main command, which is a call to the `de.saar.coli.amtools.decomposition.SourceAutomataCLI` class (`SourceAutomataCLIAMR` for AMR; have a look at `createAMRAutomata.sh` for its use). This will, in the output folder (i.e. the `-o` option; **if you reproduce the experiment, please use different folders here**), create a `train.zip` and a `dev.zip`, which contain all the information needed to train the neural parser while jointly learning the source names, in particular all tree automata and an `amconll` file with the sentences (note that the command also puts a log file into the same folder).

If you want to reproduce this (and I assume you don't want to do this for all sources and graphbanks), I recommend you just run that java command directly rather than using the script, to avoid accidentally overwriting something. I hope all the parameters are self-explanatory. Here is the content of `storeFilePaths.sh` so you can get the paths more easily:

```
export DM_TRAIN="/proj/irtg/sempardata/sdp/2015/train.dm.sdp"
export PAS_TRAIN="/proj/irtg/sempardata/sdp/2015/train.pas.sdp"
export PSD_TRAIN="/proj/irtg/sempardata/sdp/2015/train.psd.sdp"
export AMR_TRAIN="/proj/irtg/sempardata/unsupervised2020/AMRtrain.corpus"
export DM_DEV="/proj/irtg/sempardata/sdp/2015/dev.dm.sdp"
export PAS_DEV="/proj/irtg/sempardata/sdp/2015/dev.pas.sdp"
export PSD_DEV="/proj/irtg/sempardata/sdp/2015/dev.psd.sdp"
export AMR_DEV="/proj/irtg/sempardata/unsupervised2020/AMRdev.corpus"
```

This takes less than an hour usually, but a few hours for 6 sources (also doesn't need the full 800G memory for lower source numbers).

### Baselines

Similar scripts exist for creating the training data for the baselines. Again in `/proj/irtg/sempardata/unsupervised2020/scripts/`, the file `runEM.sh` creates the data for the EM baseline, `runRandom.sh` for the random tree baseline and `runEM0.sh` for the random weights baseline. These scripts create the data for all three SDP corpora; `createAMRBaselines.sh` creates the data for all three baselines for AMR. Similar notes as for the neural joint learning training data apply, except that these scripts create in the output folders not zip files, but `train.amconll` and `dev.amconll` files for supervised training.

### AMR input file

The AMR decomposition scripts want as input the preprocessed AMR corpus. For this, use the `preprocess-no-baseline.sh` script of the original `am-parser`, i.e. https://github.com/coli-saar/am-parser/blob/master/scripts/preprocess-no-baseline.sh (this may be renamed to `preprocessAMR.sh` in the course of Pauline's cleanup work). Usage see here: https://github.com/coli-saar/am-parser/wiki/Converting-individual-formalisms-to-AM-CoNLL#amr

We are then interested in the `namesDatesNumbers_AlsFixed_sorted.corpus` files in the `data/alto/train/` and `data/alto/dev/` folders of the output folder of that preprocessing script. These are the `AMRtrain.corpus` and `AMRdev.corpus` files in `/proj/irtg/sempardata/unsupervised2020/` (see also the `/proj/irtg/sempardata/unsupervised2020/readme.txt`).

## Neural parser

For the neural parser, use the `unsupervised2020` branch of `am-parser` commit `16e50ee`, or the code in `/proj/irtg/sempardata/unsupervised2020/am-parser`

### Neural joint learning (unsupervised)

For this, have a look at the scripts in `/proj/irtg/sempardata/unsupervised2020/am-parser/scripts/unsupervised2020/` (or actually https://github.com/coli-saar/am-parser/tree/unsupervised2020/scripts/unsupervised2020). They run commands of essentially this shape:

```
python -u train.py jsonnets/unsupervised2020/automata/DMallAutomaton.jsonnet
-s /local/jonasg/unsupervised2020/temp/DMAuto3-allAutomaton-jan20/ -f --file-friendly-logging
-o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/DM/train.zip"]], "validation_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/DM/dev.zip"]]}'
--comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags DM allAutomaton3 --project unsupervised2020
&> /proj/irtg/sempardata/unsupervised2020/logs/DMAuto3-allAutomaton-jan20.log &
```

(this example coming from https://github.com/coli-saar/am-parser/tree/unsupervised2020/scripts/unsupervised2020/dm3pas3psd4amr4-jan15.sh). I added some linebreaks for readability here; this should all be one line in practice. This command has several components:

1. `python -u train.py` the basic python command
2. `jsonnets/unsupervised2020/automata/DMallAutomaton.jsonnet` the AllenNLP config file with all the information about the model. Use `XallAutomaton.jsonnet` with `X` from `AMR, DM, PAS, PSD`.
3. `-s /local/jonasg/unsupervised2020/temp/DMAuto3-allAutomaton-jan20/` is the output path where the trained model and generated `amconll` files are created (the former for the best epoch, the latter for each epoch). There are two types: the first is e.g. `DM_amconll_list_train_epoch99.amconll` (and the same for dev) contain trees based on the viterbi tree of the tree automaton (for the sentences in the `train.zip` and `dev.zip` respectively). The second type is e.g. `dev_epoch_99.amconll` which contains predictions (unrestricted by the tree automaton, i.e. may yield the wrong graph) on the whole dev set. **If you reproduce the experiments, please use new output folders** (you probably don't have write permission there anyway). Comet.ml stores the file locations and servers for each experiment in the Hyperparameters tab, e.g. https://www.comet.ml/jgroschwitz/unsupervised2020-amr/2caf2b69547d4970b9e02ebee1515633?experiment-tab=params
3. `-f --file-friendly-logging` don't worry about this
4. `-o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/DM/train.zip"]], "validation_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/DM/dev.zip"]]}'` this overwrites some of the `.jsonnet` config file. I use it to (a) specify which GPU to use (c.f. https://github.com/coli-saar/am-parser/wiki/Training-the-Parser), and (b) give it the paths to the input data (here, the DM tree automata with 3 sources). If I want test set evaluation, I also set that up here (see e.g. `test-allAutomaton-dm3pas3psdOld4amr3-jan23.sh`)
5. `--comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags DM allAutomaton3 --project unsupervised2020` this sets up comet.ml logging. the number/letter sequence is my personal sequence, meaning it logs experiments into my account. I also specify which project (use `unsupervised2020-amr` for AMR, or your own). It also adds experiment-specific tags, see below.
6. `&> /proj/irtg/sempardata/unsupervised2020/logs/DMAuto3-allAutomaton-jan20.log &` just logging things (**again, change this path please when reproducing**) and adding the `&` in the script in the end to have multiple commands run in parallel.

**Tags:**

* I used the tags `DM` and `PAS` for the two graphbanks. For PSD, i used the `oldPre` tag (see PSD pre/postprocessing below), and for AMR I used no extra tag since it is a separate comet project anyway.
* I used the `allAutomatonX` tag with X the number of sources for the joint neural method. For the baselines, I used the tags `randomX`, `EM_0iterX` and `EMX`.
* For the runs that were also evaluated on the test set, I used the tags `testa`, `testb` and `testc` for the three runs.

The results are also in this spreadsheet: https://docs.google.com/spreadsheets/d/1nBs_88iLZliUK2VPq1flDhRhVuDv4D899KPdvbMJUyA/edit?usp=sharing


### Baselines (supervised)

Training the baselines is similar, but uses different jsonnet files (since the neural architecture is different for supervised training) and of course the respective amconll files as input, not the zip files with the automata. The relevant scripts are:

* https://github.com/coli-saar/am-parser/blob/unsupervised2020/scripts/unsupervised2020/trainSDP_EM3.sh
* https://github.com/coli-saar/am-parser/blob/unsupervised2020/scripts/unsupervised2020/trainSDP_EM_0iter3.sh
* https://github.com/coli-saar/am-parser/blob/unsupervised2020/scripts/unsupervised2020/trainSDP_random3.sh
* https://github.com/coli-saar/am-parser/blob/unsupervised2020/scripts/unsupervised2020/amrEM3-iter0-rnd-retrain3.sh

with the last one also running a 'retrain' experiment (supervised training on a training file generated by the neural joint learning model) that didn't make it into the paper.

## Variants

### 6 sources

Running the joint neural learning with 6 sources takes extra memory. I therefore created `/proj/irtg/sempardata/unsupervised2020/am-parser-bigmemory/`, which is an am-parser clone but in `graph_dependency_parser/important_imports.py` gives Pyjnius 150G instead of 50G memory. Also batch sizes are reduced to 24 (AMR 16), and evaluation starts at epoch 10 (AMR 14) since training is slower.

### AMR all edges (not in paper)

I also ran the AMR experiments without the step in preprocessing that removes coreference edges based on the old AM decomposition heuristics. Everything with `allEdges` or `AllEdges` refers to this. To obtain the training data, I ran the same commands as in https://github.com/coli-saar/am-parser/blob/master/scripts/preprocess-no-baseline.sh, but only up to the calls of `de.saar.coli.amrtagging.formalisms.amr.tools.datascript.RawAMRCorpus2TrainingData` (the rest is not necessary for any experiment here), and I ran these calls without the `--corefSplit` flag (note that this only works in the `new_decomposition` branch at this point, since previously this flag was bugged and always defaulted to true, whether given or not).

### PSD pre/postprocessing (not in paper)

There are three versions of the PSD conjunction pre- and postprocessing: (a) none, (b) as in ACL19 ('old') and (c) new (which also performs the pre/postprocessing for modifier edges). I used (b) in the paper since it is the easiest to explain ('same as ACL19'), is comparable to ACL19 and had best performance. For this, use the `--useLegacyPSDpreprocessing` option in `de.saar.coli.amtools.decomposition.SourceAutomataCLI`. For (a), use the `--noPSDpreprocessing` option in `SourceAutomataCLI` and in https://github.com/coli-saar/am-parser/blob/unsupervised2020/configs/eval_commands.libsonnet (or rather your local copy) replace in "commands" whatever is after "PSD" with "self.DM" (like it is for PAS). For (c), use no extra option in `SourceAutomataCLI`. In the evaluation commands, the difference between (b) and (c) is the `--legacyACL19` option of `ToSDPCorpus` (only available in the master branch). 

**TODO** I may have accidentally not used the `--legacyACL19` option in our experiments. I seem to remember that it didn't make a difference in that direction anyway (since the additional preprocessing didn't happen, trying to do additional postprocessing simply doesn't do anything), but I should double check for the camera ready version.

### Supervised edge existance and lexical label (not in paper)

To use supervised loss for edge existence and lexical labels (since they are given even if the source names are not), use the `Xautomaton.jsonnet` files with `X` from `AMR, DM, PAS, PSD`. Comet experiments for this used the tag `autoS` where S is the number of sources. Note that this worked worse for AMR but similar to the `allAutomaton` method for SDP.


