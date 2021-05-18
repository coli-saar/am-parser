
Experiments on 
the [COGS](https://github.com/najoungkim/COGS) dataset 
of [Kim & Linzen (2020)](https://www.aclweb.org/anthology/2020.emnlp-main.731/).

**Note**: This README might get moved to the am-parser wiki at some point.

## Steps:

1. Download the am-parser and checkout the `cogs_unsupervised` branch.
Install necessary packages and dependencies (see the main README)

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
   1. Get the correct `am-tools.jar` that can deal with COGS (on the branch mentioned above!).
   With intelliJ: gradle >  am-tools > Tasks > shadow > shadowJar > Run   and copy the resulting file (`am-tools/build/libs/am-tools.jar`) )
   Place the jar-file in the `am-parser` folder.
   2. Call `de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS` to get the zip-files with train and dev data:
   You have to provide path to the train and dev corpora as input, an output path, the number of sources you want,
   the algorithm option should be `automata`.  
   **todo** below command not tested, find out what's working
```bash
java -cp am-tools.main de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS -t COGSDATADIR/train_noprim.tsv -d COGSDATADIR/dev.tsv -o ../cogs2021/amconll/ -nrSources 3 --algorithm automata
```
5. **to do** a config file, e.g. `am-parser/jsonnets/cogs2021/firsttry.jsonnet` 
   - train on train or train100?
   - number of sources? default is 3?
   - use BERT or learn embeddings from cogs data alone ('tokens')?
6. Train [see also wiki of am-parser](https://github.com/coli-saar/am-parser/wiki/Training-the-Parser)
```bash
python -u am-parser/train.py <config-file> -s <where to save the model>  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  <your cuda device>  } }' &> <where to log output>
```
or
```bash
./firsttry_train.sh
```
7. Evaluate on test data [see also wiki of am-parser](https://github.com/coli-saar/am-parser/wiki/Prediction-and-evaluation-on-test-data)
```bash
bash am-parser/scripts/predict.sh -i COGSDATADIR/test.tsv -T COGS -o <output directory> -m <your model>
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

