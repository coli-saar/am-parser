
Experiments on 
the [COGS](https://github.com/najoungkim/COGS) dataset 
of [Kim & Linzen (2020)](https://www.aclweb.org/anthology/2020.emnlp-main.731/).

**Note**: This README might get moved to the am-parser wiki at some point.

## Steps:

1. Download the am-parser and checkout the `cogs_unsupervised` branch.
   Install necessary packages and dependencies (see the main README).
   Note: we need a special `am_tools.jar`, see below.
   

2. Get the COGS corpus
```bash
git clone https://github.com/najoungkim/COGS.git
```
   In what follows we will use `COGSDATADIR` to refer to 
   the path to the `data` subdirectory of this repository.

3. (Pre-experiments w/o primitives **to do: get rid of this restriction** ) 
   Strip all primitives (in train, train100) bc so far pre/postprocessing 
   can't deal with some of them
```bash
python3 am-parser/scripts/cogs2021/stripprimitives.py --cogsdatadir COGSDATADIR
```
   where `COGSDATADIR` is the path to the `data` subdirectory of the COGS repository (e.g. `../cogs2021/COGS/data/`).

4. use the ([`cogs_new_decomp` branch](https://github.com/coli-saar/am-tools/tree/cogs_new_decomp) of am-tools:
   After the sub-steps below your `AMCONLLDIR` (e.g. `../cogs2021/amconll/`) should contain a `train.zip`, `dev.zip` and a `dev_dp.amconll` file.
   1. Get the correct `am-tools.jar` that can deal with COGS (on the branch mentioned above!).
   With IntelliJ: `gradle >  am-tools > Tasks > shadow > shadowJar > Run` and 
   copy the resulting file (`am-tools/build/libs/am-tools.jar`) into the `am-parser` folder.
   2. Call `de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS` to get the zip-files with train and dev data:
   You have to provide 
   - the path to the train and dev corpora as input (e.g. `-t COGSDATADIR/train_noprim.tsv -d COGSDATADIR/dev.tsv`)
   - an output path (e.g. `-o AMCONLLDIR`)
   - the number of sources you want (e.g. `-nrSources 3`)
   - the algorithm option should be `--algorithm automata`.  
   **todo** below command not tested, find out what's working
```bash
java -cp am-tools.main de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS -t COGSDATADIR/train_noprim.tsv -d COGSDATADIR/dev.tsv -o AMCONLLDIR -nrSources 3 --algorithm automata
```
   3. Prepare dev data for evaluation:  
      the produced amconll file is used as `system_input` for the `validation_evaluator`.
      Note: the `--corpus` should be the same as the dev corpus above
      (In principle you could use an output directory different from the previous stepm but we won't do this here)
      **todo below command not tested**
```bash
java -cp am-tools.main de.saar.coli.amrtagging.formalisms.cogs.tools.PrepareDevData --corpus COGSDATADIR/dev.tsv --outPath AMCONLLDIR --prefix dp_dev
```
5. **to do** a config file, e.g. `am-parser/jsonnets/cogs2021/debugging.jsonnet` 
   make sure the file paths in the config file (and also imported libsonnet files) 
   work for you and contain the files you are interested in 
   (e.g. `COGSDATADIR` and `AMCONLLDIR` and corresponding filenames in the folders).
   - train on train or train100?
   - number of sources? default is 3?
   - use BERT or learn embeddings from cogs data alone ('tokens')?
6. **Training** ([see also wiki of am-parser](https://github.com/coli-saar/am-parser/wiki/Training-the-Parser)).
   Let's assume its files (`metrics.json`, `model.tar.gz` among others) should 
   be written to some folder `MODELFOLDER` (e.g. `../cogs2021/temp`) 
   and you have one GPU (cuda device 0).
```bash
cd am-parser
python -u train.py jsonnets/cogs2021/CONFIGFILE.jsonnet -s MODELFOLDER/  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  0  } }' &> ./train_cogs.log
```
   (pw: `./debugging_train.sh`)
   While training you can do `tensorboard --logdir=MODELFOLDER` to check model progress 
   (important: last slash in path of MODELFOLDER should be omitted here).
   Or you use [comet.ml](https://www.comet.ml/site/).

7. **Evaluate on test data** ([see also wiki of am-parser](https://github.com/coli-saar/am-parser/wiki/Prediction-and-evaluation-on-test-data)).
   Assume you want to store the output in some `OUTPUTDIR` (e.g. `../cogs2021/output`),
   then the prediction will create 3 files in this folder:
   - `COGS_gold.amconll` gold AM dependency trees
   - `COGS_pred.amconll` predictions as AM dependency trees
   - `COGS.tsv` predictions in the native TSV format of COGS
   **to do: modifications not pushed yet** 
   I had to modify `predict.sh`
   (add `COGS` as allowed formalism, 
   add respective lines to prepare the dev data (`PrepareDevData`) and 
   to transform predictions back to COGS format including running evaluation (`ToCOGSCorpus`))
   and `parse_file.py` (adaptions for the automata/unsupervised case) to get it to work.
```bash
bash am-parser/scripts/predict.sh -i COGSDATADIR/test.tsv -T COGS -o OUTPUTDIR -m MODELFOLDER/model.tar.gz -g 0
```
   **todo** question to pauline: do I really have to build EDM each time even when evaluating on a corpus different from EDS?


pw:
```bash
/bin/bash /home/wurzel/HiwiAK/am-parser/scripts/cogs2021/debugging_train.sh
bash ./scripts/predict.sh -i ../cogs2021/small/test5.tsv -T COGS -o ../cogs2021/output -m ../cogs2021/temp/model.tar.gz -g 0 -f &> ../cogs2021/predict-sh.log
```


## experiments:

different factors:
- training set (`train` or `train100`)
- embeddings: from BERT (`bert`), learnt from cogs alone (`nobert`)
- number of sources: 3,4,5?? **todo**

Format of jsonnet filenames: **todo**

## further notes

The COGS dataset comes with a training set (`train`: 24,155 samples), 
a bigger training set (`train_100`: 39,500), a `dev` and `test` set (3k each),
and a generalization set (`gen`: 21k samples, 1k per generalization type).

Note: The COGS dataset itself doesn't contain graphs, but logical forms.
Therefore, we need a pre/postprocessing step (graph conversion)
which is implemented in [am-tools (branch `cogs_new_decomp`)](https://github.com/coli-saar/am-tools/tree/cogs_new_decomp)).  
**Important:** the `am-tools.jar` that is used here must be created from the correct am-tools branch, 
otherwise the parser wouldn't know how to do the pre-/post-processing 
(and also evaluation metrics are calculated during postprocessing).

Note: currently, all this is based on the 'unsupervised' parser idea of JG 
(see also the respective branches in am-parser and am-tools): that is, source names are learnt!

