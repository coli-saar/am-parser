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

It first does some setup, describing output path and the path to the jar file, and loading input filepaths on the server from `storeFilePaths.sh`. Then it prints and evaluates the main command, which is a call to the `de.saar.coli.amtools.decomposition.SourceAutomataCLI` class (`SourceAutomataCLIAMR` for AMR; have a look at `createAMRAutomata.sh` for its use). This will, in the output folder (i.e. the `-o` option), create a `train.zip` and a `dev.zip`, which contain all the information needed to train the neural parser while jointly learning the source names, in particular all tree automata and an `amconll` file with the sentences (note that the command also puts a log file into the same folder).

### Baselines

Similar scripts exist for creating the training data for the baselines. Again in `/proj/irtg/sempardata/unsupervised2020/scripts/`, the file `runEM.sh` creates the data for the EM baseline, `runRandom.sh` for the random tree baseline and `runEM0.sh` for the random weights baseline. The scripts create the data for all three SDP corpora; `createAMRBaselines.sh` creates the data for all three baselines for AMR. Similar notes as for the neural joint learning training data apply, except that these scripts create in the output folders not zip files, but `train.amconll` and `dev.amconll` files for supervised training.

### AMR input file

The AMR decomposition scripts want as input the preprocessed AMR corpus. For this, use the `preprocess-no-baseline.sh` script of the original `am-parser`, i.e. https://github.com/coli-saar/am-parser/blob/master/scripts/preprocess-no-baseline.sh (this may be renamed to `preprocessAMR.sh` in the course of Pauline's cleanup work). Usage see here: https://github.com/coli-saar/am-parser/wiki/Converting-individual-formalisms-to-AM-CoNLL#amr

We are then interested in the `namesDatesNumbers_AlsFixed_sorted.corpus` files in the `data/alto/train/` and `data/alto/dev/` folders of the output folder of that preprocessing script. This are the `AMRtrain.corpus` and `AMRdev.corpus` files in `/proj/irtg/sempardata/unsupervised2020/` (see also the `/proj/irtg/sempardata/unsupervised2020/readme.txt)`).

## Neural parser



### Neural joint learning (unsupervised)



### Baselines (supervised)
