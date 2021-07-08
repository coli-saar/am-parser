
Experiments on 
the [COGS](https://github.com/najoungkim/COGS) dataset 
of [Kim & Linzen (2020)](https://www.aclweb.org/anthology/2020.emnlp-main.731/).

**Note**: This README might get moved to the am-parser wiki at some point.

## Steps:

### prerequisites: am-parser, am-tools, COGS corpus

#### (a) Set up the conda environment and am-parser install
Use the conda environment `allennlp` as described in the [am-parser wiki](https://github.com/coli-saar/am-parser/wiki/Setup-and-file-locations-on-the-Saarland-servers#first-time-setup).
Clone the am-parser repository and make sure to checkout the ['cogs_unsupervised' branch](https://github.com/coli-saar/am-parser/tree/cogs_unsupervised)
as this is the branch containing the relevant code.
Install necessary packages and dependencies for the am-parser (see the main README).
Note: we need a special `am_tools.jar`, see the next point.
We will use `AMPARSERDIR` to refer to directory path of the `am-parser`.
- local (pw): use `am-parser` conda environment instead, am-parser at `~/HiwiAK/am-parser`
- coli servers: the am-parser directory can be found at `x` (commit `x`) (**to do: location/commit on server**)

#### (b) getting the right am-tools.jar file
The code for COGS can be found in the ['cogs_new_decomp' branch of am-tools](https://github.com/coli-saar/am-tools/tree/cogs_new_decomp).
To get the jar file run the `shadowJar` task of gradle 
(IntelliJ: `gradle >  am-tools > Tasks > shadow > shadowJar > Run`) 
and the jar file will be generated at `am-tools/build/libs/am-tools.jar`.
Copy this jar file into the am-parser folder.
On the servers the am-tools.jar file was produced using commit `x` of am-tools. (**to do: location/commit on server**).
Important: If the am-parser can't find an am-tools.jar file it might try to download it, but this won't be the version we need for the COGS experiments! 

#### (c) download COGS data
The COGS data is available as a [GitHub repository](https://github.com/najoungkim/COGS).
```bash
git clone https://github.com/najoungkim/COGS.git
```
The `data` subdirectory contains the corpora files (TSV format).
In what follows we will sometimes use `COGSDATADIR` to refer to the path to the `data` subdirectory of this repository.
- local (pw): `~/HiwiAK/cogs2021/COGS/data`
- coli servers: the directory can be found at `/proj/irtg/sempardata/cogs2021/data/COGS/data/` (commit `6f663835897945e94fd330c8cbebbdc494fbb690`) (**to do: change permissions**).

#### (d) preliminary: strip primitives
(Pre-experiments w/o primitives **to do: get rid of this restriction** ) 
Strip all primitives (in train, train100) because so far pre/postprocessing can't deal with some of them.
```bash
cd AMPARSERDIR
python3 scripts/cogs2021/stripprimitives.py --cogsdatadir COGSDATADIR
```
where `COGSDATADIR` is the path to the `data` subdirectory of the COGS repository.
- local (pw): `python3 scripts/cogs2021/stripprimitives.py --cogsdatadir ~/HiwiAK/cogs2021/COGS/data/`



### Step 1: Decomposition and graph conversion (using am-tools)

The conversion COGS logical forms to graphs and back is implemented in am-tools,
the conversion to graphs is a preprocessing step, the conversion from graphs a postprocessing step.
After the preprocessing, we need to decompose the graphs into blobs and infer an AM dependency structure.
Because we use the 'unsupervised' version here (learning source names) the result of the decomposition is not an AMCONLL file,
but zip files containing tree automata.

Remember we need the right am-tools.jar file (from the [cogs_new_decomp branch](https://github.com/coli-saar/am-tools/tree/cogs_new_decomp))!
In this step we will create files which we will put into a new directory.
Create a new directory and let's call the directory path `AMCONLLDIR`.
  - local (pw): `~/HiwiAK/cogs2021/amconll/`
  - coli servers: **to do: location on coli servers**, e.g. `/proj/irtg/sempardata/cogs2021/first_experiments/auto3/training_input/`

After you performed the two sub-steps (a,b) below, your `AMCONLLDIR` should contain 
  - zip files for train and dev set (`train.zip`, `dev.zip`), and 
  - an amconll file of the dev set for validation (probably called `dev_dp.amconll`).
The following two points (a,b) can be performed with the `get_train_dev.sh` script.
```bash
cd AMPARSERDIR
bash ./scripts/cogs2021/get_train_dev.sh -t COGSDATADIR/train_nonprim.tsv -d COGSDATADIR/dev.tsv -o AMCOLLDIR -s 3 -p dp_dev
# actually `-s 3` (3 sources) and `-p dp_dev` (file dp_dev.amconll will be created) are defaults, so without them it's the same:
# bash ./scripts/cogs2021/get_train_dev.sh -t COGSDATADIR/train_nonprim.tsv -d COGSDATADIR/dev.tsv -o AMCOLLDIR
```
- pw debugging: `/bin/bash ~/HiwiAK/am-parser/scripts/cogs2021/get_train_dev.sh -t ~/HiwiAK/cogs2021/small/train20_nonprim.tsv -d ~/HiwiAK/cogs2021/small/dev10.tsv -o ~/HiwiAK/cogs2021/amconll/`
On the coli servers e.g. (maybe adjust experiment folder and whether nonprim version :
```bash
cd AMPARSERDIR
bash get_train_dev.sh -t /proj/irtg/sempardata/cogs2021/data/COGS/data/train_nonprim.tsv -d /proj/irtg/sempardata/cogs2021/data/COGS/data/dev.tsv -o /proj/irtg/sempardata/cogs2021/first_experiments/auto3/training_input/ -s 3 -p dp_dev
# 3 sources, dp_dev.amconll created
```

#### (a) Getting zip files: `SourceAutomataCLICOGS`
The input are the TSV files of COGS, output are zip files.
Call `de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS` with the right command line options.
You have to provide 
- the path to the train and dev corpora as input (e.g. `-t COGSDATADIR/train_noprim.tsv -d COGSDATADIR/dev.tsv`)
- an output path (e.g. `-o AMCONLLDIR`)
- the number of sources you want (e.g. `--nrSources 3`)
- the algorithm option should be `--algorithm automata`.  
```bash
cd AMPARSERDIR
java -cp am-tools.jar de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS -t COGSDATADIR/train_noprim.tsv -d COGSDATADIR/dev.tsv -o AMCONLLDIR --nrSources 3 --algorithm automata
```


#### (b) Getting amconll files for validation: `PrepareDevData`
the produced amconll file is used as `system_input` for the `validation_evaluator`.
Note: the `--corpus` should be the same as the dev corpus above.
Specifically note the prefix option.
(In principle you could use an output directory different from the previous step, but we won't do this here)
```bash
cd AMPARSERDIR
java -cp am-tools.jar de.saar.coli.amrtagging.formalisms.cogs.tools.PrepareDevData --corpus COGSDATADIR/dev.tsv --outPath AMCONLLDIR --prefix dp_dev
```


### Step 2: Training the parser (using am-parser)

#### (a) Pick a config file
e.g. `AMPARSERDIR/jsonnets/cogs2021/debugging.jsonnet`.   
Make sure the file paths in the config file (and also imported libsonnet files) 
work for you and contain the files you are interested in 
(e.g. `COGSDATADIR` and `AMCONLLDIR` and corresponding filenames in the folders).
**to do** note on available config files: what do their names mean?
- training on train or train100?
- number of sources? default is 3?
- use BERT or learn embeddings from cogs data alone ('tokens')?
- other factors?

#### (b) Actual training
([see also wiki of am-parser](https://github.com/coli-saar/am-parser/wiki/Training-the-Parser)).
Let's assume its files (`metrics.json`, `model.tar.gz` among others) should 
be written to some folder `MODELDIR` (e.g. `../cogs2021/temp`), 
you have one GPU (cuda device 0),
and log output should be written to LOGFILE.
```bash
cd AMPARSERDIR
python -u train.py jsonnets/cogs2021/CONFIGFILE.jsonnet -s MODELDIR/  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  0  } }' &> LOGFILE
```
**to do: add server version and comet, more options?**
- notes for pw local debugging:
  - I had to `LC_ALL=en_US.UTF-8` as with my German local a learning rate of `0.01` is interpreted as just `0`.
  - `./debugging_train.sh`

*Monitoring training progress*  
While training you can do `tensorboard --logdir=MODELDIR` to check model progress 
(important: last slash in path of MODELDIR should be omitted here).
Or you use [comet.ml](https://www.comet.ml/site/).

*How long does training take?*  
Training time obviously depends on many factors such as 
- the number of epochs, batch size, size of the neural model and other hyperparameters
- the corpora used (full corpus? train or train100?)
- the hardware (GPU?)

Here are some examples: **to do: training times**
- local (pw): just minutes on a small train/dev (20-nonprim, 10dev) for a small debugging model (32 word dim, 32/64 hidden, k=4 supertag decoding, batch size 1, no dropout) with early stopping (100 epochs, 20 patience).
- coli servers: GPU? config file/epochs/train set...

**to do: maybe make a spreadsheet with results and link it here?**


### Step 3: (optional) predictions (using am-parser) and evaluation

#### (a) Get predictions using a trained model
Note that the training must be successfully completed (you should have a `model.tar.gz` in your `MODELDIR`).
Assume you want to store the output in some `OUTPUTDIR` (e.g. `../cogs2021/output`),
then the prediction will create 3 files in this folder:
- `COGS_gold.amconll` gold AM dependency trees
- `COGS_pred.amconll` predictions as AM dependency trees
- `COGS.tsv` predictions in the native TSV format of COGS

Run the following assuming that 
- you have one GPU available (`-g 0`) 
- you would like to save logs to `EVALLOGFILE`
- you would like to generate predictions for the `test.tsv` file
```bash
cd AMPARSERDIR
bash ./scripts/cogs2021/unsupervised_predict.sh -i COGSDATADIR/test.tsv -o OUTPUTDIR -m MODELDIR/model.tar.gz -g 0 &> EVALLOGFILE
```
Note: you could add the `-f` option for fast
- local (pw): `bash ./scripts/cogs2021/unsupervised_predict.sh -i ../cogs2021/small/test5.tsv -o ../cogs2021/output -m ../cogs2021/temp/model.tar.gz -g 0 -f &> ../cogs2021/predict-sh.log`
- coli severs, eg. `bash ./scripts/cogs2021/unsupervised_predict.sh -i /proj/irtg/sempardata/cogs2021/data/COGS/data/test.tsv -o /proj/irtg/sempardata/cogs2021/first_experiments/auto3/predictions -m /proj/irtg/sempardata/cogs2021/first_experiments/auto3/training_output/model.tar.gz -g 0 -f &> /proj/irtg/sempardata/cogs2021/first_experiments/auto3/predict-sh.log`

(see also [the am-parser wiki on prediction and evaluation on test data](https://github.com/coli-saar/am-parser/wiki/Prediction-and-evaluation-on-test-data),
but due to using the 'unsupervised' approach and a new formalism COGS might not be applicable).

#### (b) Computing evaluation metrics
The COGS authors use exact match accuracy as the main evaluation metric and
reported overall results as well as result per generalization type (see 3rd column in the TSV).
In their appendix they also mention token-level edit distance and ill-formedness (closing parenthesis missing).
As of now (May 2021) there doesn't seem to be a separate evaluation script.
(first author told me a while ago they computed their values using OpenNMT which they also used to train their models).
**to do: which eval script used? incl generalization type specific results?**


### pw notes local debugging

- don't forget to copy most recent am-tools.jar into am-parser directory!
- don't forget to change the file paths in config files and bash scripts 
(am-tools train-dev prepare, jsonnet files, command line arguments for train and predict)
- also if you re-run the pipeline, make sure that errors are not hidden by accidentally using a file created in the last run.
- commands:
```bash
cd ~/HiwiAK/am-parser/
bash ./scripts/cogs2021/get_train_dev.sh -t ../cogs2021/small/train20_nonprim.tsv -d ../cogs2021/small/dev10.tsv -o ../cogs2021/amconll/ -s 3 -p dp_dev
bash ./scripts/cogs2021/debugging_train.sh
bash ./scripts/cogs2021/unsupervised_predict.sh -i ../cogs2021/small/test5.tsv -o ../cogs2021/output -m ../cogs2021/temp/model.tar.gz -g 0 -f &> ../cogs2021/predict-sh.log
# bash ./scripts/predict.sh -i ../cogs2021/small/test5.tsv -T COGS -o ../cogs2021/output -m ../cogs2021/temp/model.tar.gz -g 0 -f &> ../cogs2021/predict-sh.log
```


A* parser
(note: astar implemented in am-tools: version on main branch is different from the one on the cogs branch!!!)
Second version: (ultimately added `-p` option to `unsupervised_predict.sh`, still model was trained with fixed-tree decoder)
```bash
bash ./scripts/cogs2021/unsupervised_predict.sh -i ../cogs2021/small/test5.tsv -o ../cogs2021/decoding/test -m ../cogs2021/temp/model.tar.gz -g 0 -p &> ../cogs2021/decoding/predict-sh.log
```
First version:
```bash
# train
bash ./scripts/cogs2021/get_train_dev.sh -t ~/HiwiAK/cogs2021/small/train50.tsv -d ~/HiwiAK/cogs2021/small/dev10.tsv -o ~/HiwiAK/cogs2021/amconll/ -s 3 -p dp_dev
bash ./scripts/cogs2021/debugging_train.sh
tensorboard --logdir=../cogs2021/temp

# see  https://github.com/coli-saar/am-parser/wiki/Computing-scores
# python dump_scores.py models/a_model <formalism> <input data.amconll> <output file.zip> --cuda-device 0
python dump_scores.py ../cogs2021/temp COGS ../cogs2021/amconll/dp_dev.amconll ../cogs2021/decoding/scores/scores.zip --cuda-device 0
# see https://github.com/coli-saar/am-parser/wiki/A*-Parser
# --outside-estimator OPTION
# --threads <N>
# --statistics <statistics.csv>   #runtime stats
# NOTE: in cogs branch am-tools.jar version: Astar is still in de.saar.coli.irtg.experimental.astar.Astar
# java -cp <am-tools.jar> de.saar.coli.amtools.astar.Astar -s <scores.zip> -o <outdir> 
java -cp am-tools.jar de.saar.coli.amtools.astar.Astar -s ../cogs2021/decoding/scores/scores.zip -o ../cogs2021/decoding/output
# -> output directory now contains log_*.txt and result_*.amconll
# speeding up computations?
# java -cp <am-tools.jar> de.saar.coli.amtools.astar.io.SerializedScoreReader <scores.zip> <serialized-scores.zip>
# compare to gold:
mv ../cogs2021/decoding/output/results_*.amconll ../cogs2021/decoding/output/results.amconll
mv ../cogs2021/decoding/output/log_*.txt ../cogs2021/decoding/output/log.txt
java -cp am-tools.jar de.saar.coli.amrtagging.formalisms.cogs.tools.ToCOGSCorpus -c ../cogs2021/decoding/output/results.amconll -o ../cogs2021/decoding/eval/COGS_pred.tsv --gold ../cogs2021/small/dev10.tsv --verbose
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
