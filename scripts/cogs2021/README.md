
Experiments on 
the [COGS](https://github.com/najoungkim/COGS) dataset 
of [Kim & Linzen (2020)](https://www.aclweb.org/anthology/2020.emnlp-main.731/).
Using the 'unsupervised' parser ([Groschwitz et al. 2021](https://aclanthology.org/2021.spnlp-1.3/))
**Note**: This README might get moved to the am-parser wiki at some point.

## Steps:

### prerequisites: am-parser, am-tools, COGS corpus

#### (a) Set up the conda environment and am-parser install

(1) Clone the am-parser repository and make sure to check out the ['cogs_unsupervised' branch](https://github.com/coli-saar/am-parser/tree/cogs_unsupervised)
as this is the branch containing the relevant code.  
(2) Install necessary packages and dependencies for the am-parser (see the main README).
On the coli-servers, you can use the conda environment `allennlp` as described in the [am-parser wiki](https://github.com/coli-saar/am-parser/wiki/Setup-and-file-locations-on-the-Saarland-servers#first-time-setup).
The environment that I used on my local computer is described in [this section below](#environment-used-locally)

(3) Note: we need a special `am_tools.jar`, see the next point.

We will use `AMPARSERDIR` to refer to directory path of the `am-parser`.
- local (pw): use `am-parser` conda environment instead, am-parser at `~/HiwiAK/am-parser`
- coli servers: the am-parser directory can be found at `/local/piaw/am-parser` on jones-2 (commit `x`) (**to do: location/commit on server**)


#### (b) getting the right am-tools.jar file

The code for COGS can be found in the ['cogs_new_decomp' branch of am-tools](https://github.com/coli-saar/am-tools/tree/cogs_new_decomp).
To get the jar file run the `shadowJar` task of gradle 
(IntelliJ: `gradle >  am-tools > Tasks > shadow > shadowJar > Run`) 
and the jar file will be generated at `am-tools/build/libs/am-tools.jar`.
Copy this jar file into the am-parser folder.
On the servers the am-tools.jar file was produced using commit `x` of am-tools. 
It can be found at `/proj/irtg/sempardata/cogs2021/jar/am-tools.jar` (**to do: location/commit on server**).
So can copy it to local: `cp /proj/irtg/sempardata/cogs2021/jar/am-tools.jar /local/piaw/am-parser/am-tools.jar`
Important: If the am-parser can't find an am-tools.jar file it might try to download it, but this won't be the version we need for the COGS experiments! 

**todo: if you would like to use the Astar parser, you might use an additional jar from the master branch**
`cp /proj/irtg/sempardata/cogs2021/jar/master-branch-jar/am-tools.jar /local/piaw/am-parser/master-am-tools.jar`

#### (c) download COGS data
The COGS data is available as a [GitHub repository](https://github.com/najoungkim/COGS).
```bash
git clone https://github.com/najoungkim/COGS.git
```
The `data` subdirectory contains the corpora files (TSV format).
In what follows we will sometimes use `COGSDATADIR` to refer to the path to the `data` subdirectory of this repository.
- local (pw): `~/HiwiAK/cogs2021/COGS/data`
- coli servers: the directory can be found at `/proj/irtg/sempardata/cogs2021/data/COGS/data/` (commit `6f663835897945e94fd330c8cbebbdc494fbb690`) (**to do: change permissions**).

#### (d) no longer needed: strip primitives

This section remains there for historically reasons (pre-experiments w/o primitives),
but you can just ignore it. 
if you would still like to use it, note that data paths in later steps would need to be changed (`train.tsv` to `train_nonprim.tsv`, and same for train100),
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
  - local (pw): `~/HiwiAK/cogs2021/toy_model_run/training_input/`
  - coli servers: **to do: location on coli servers**, e.g. `/proj/irtg/sempardata/cogs2021/first_experiments/auto3prim/inputs/train/` for train.tsv and or `/proj/irtg/sempardata/cogs2021/first_experiments/auto3prim/inputs/train100/` for `train_100.tsv`

After you performed the two sub-steps (a,b) below, your `AMCONLLDIR` should contain 
  - zip files for train and dev set (`train.zip`, `dev.zip`), and 
  - an amconll file of the dev set for validation (probably called `dp_dev.amconll`)
  - (when option `-e` provided) a `test.amconll` file.

The following two points (a,b) can be performed with the `get_train_dev.sh` script.
```bash
cd AMPARSERDIR
bash ./scripts/cogs2021/get_train_dev.sh -t COGSDATADIR/train.tsv -d COGSDATADIR/dev.tsv -o AMCONLLDIR -s 3 -p dp_dev
# actually `-s 3` (3 sources) and `-p dp_dev` (file dp_dev.amconll will be created) are defaults, so without them it's the same:
# bash ./scripts/cogs2021/get_train_dev.sh -t COGSDATADIR/train.tsv -d COGSDATADIR/dev.tsv -o AMCONLLDIR
```
Note: You can add the `-r` option for preposition reification.
Note: You can add `-e COGSDATADIR/test.tsv` if you would like to evaluate a model on test directly after training (when jsonnet file contains `"evaluate_on_test" : true,` and a valid `test_evaluator`)

- pw debugging:
```bash
bash ./scripts/cogs2021/get_train_dev.sh -t ~/HiwiAK/cogs2021/small/train50.tsv -d ~/HiwiAK/cogs2021/small/dev10.tsv -o ~/HiwiAK/cogs2021/toy_model_run/training_input/ -e ~/HiwiAK/cogs2021/small/test50.tsv -r -s 3 -p dp_dev &> ../cogs2021/toy_model_run/get_train_dev.log
```
Note: this includes option `-r` for preposition reification and creating a `test.amconll` which can be used for test evaluation directly after training the model.
- On the coli servers:
```bash
# with 3 sources and dp_dev.amconll created: For train.tsv or train_100.tsv respectively
cd AMPARSERDIR
bash ./scripts/cogs2021/get_train_dev.sh -t /proj/irtg/sempardata/cogs2021/data/COGS/data/train.tsv -d /proj/irtg/sempardata/cogs2021/data/COGS/data/dev.tsv -o /proj/irtg/sempardata/cogs2021/first_experiments/auto3prim/inputs/train/ -s 3 -p dp_dev
bash ./scripts/cogs2021/get_train_dev.sh -t /proj/irtg/sempardata/cogs2021/data/COGS/data/train_100.tsv -d /proj/irtg/sempardata/cogs2021/data/COGS/data/dev.tsv -o /proj/irtg/sempardata/cogs2021/first_experiments/auto3prim/inputs/train100/ -s 3 -p dp_dev
```
Note: add `-r` for preposition reification and `-e /proj/irtg/sempardata/cogs2021/data/COGS/data/test.tsv` for creating a `test.amconll` that can be used for test evaluation after training the model.

#### (a) Getting zip files: `SourceAutomataCLICOGS`
The input are the TSV files of COGS, output are zip files.
Call `de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS` with the right command line options.
You have to provide 
- the path to the train and dev corpora as input (e.g. `-t COGSDATADIR/train.tsv -d COGSDATADIR/dev.tsv`)
- an output path (e.g. `-o AMCONLLDIR`)
- the number of sources you want (e.g. `--nrSources 3`)
- the algorithm option should be `--algorithm automata`.  
```bash
cd AMPARSERDIR
java -cp am-tools.jar de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS -t COGSDATADIR/train.tsv -d COGSDATADIR/dev.tsv -o AMCONLLDIR --nrSources 3 --algorithm automata
```
Note: You can add the ` --reifyprep` flag to enable preposition reification

#### (b) Getting amconll files for validation: `PrepareDevData`
the produced amconll file is used as `system_input` for the `validation_evaluator`.
Basically it creates an 'empty' amconll file in the sense that it only contains the words and dummy values for all other columns (so no gold data from the target side it exposed).
Note: the `--corpus` should be the same as the dev corpus above.
Specifically note the prefix option.
(In principle you could use an output directory different from the previous step, but we won't do this here)
```bash
cd AMPARSERDIR
java -cp am-tools.jar de.saar.coli.amrtagging.formalisms.cogs.tools.PrepareDevData --corpus COGSDATADIR/dev.tsv --outPath AMCONLLDIR --prefix dp_dev
```
By the way, there are two more scenarios for which you need a `PrepareDevData` call:
1. test set used as the `--corpus` argument: needed when `evaluate_on_test : true` in the jsonnet training file
2. prediction on the gen set: we need an 'empty' amconll file for prediction on the generalization set, and we get it by running `PrepareDevData` with the gen ste as the `--corpus` argument

### Step 2: Training the parser (using am-parser)

#### (a) Pick a config file
e.g. `AMPARSERDIR/jsonnets/cogs2021/debugging.jsonnet` or `AMPARSERDIR/jsonnets/cogs2021/COGSallAutomaton.jsonnet`   
Make sure the file paths in the config file (and also imported libsonnet files) 
work for you and contain the files you are interested in 
(e.g. `COGSDATADIR` and `AMCONLLDIR` and corresponding filenames in the folders).
Or you might want to override the specific key-value pairs 
(`train_data_path`, `validation_data_path`, `validation_evaluator : system_input`, `validation_evaluator : gold_file`) 
in the config file when calling the train script.
**to do** note on available config files: what do their names mean?
- tied to `get_train_dev`, so dependent on the input files of training:
  - training on train or train100? (look at `path_prefix`, and the `*_corpus_path` variables)
  - prep reification: yes or no?
  - number of sources: used 3 mostly
- use BERT or learn embeddings from cogs data alone ('tokens')? (look at `embedding_name` variable)
- supervised loss for edge existence and lex label (`all_automaton_loss : false`)
- other factors? (learning rate, number of epochs, patience, and so on)

#### (b) Actual training
([see also wiki of am-parser](https://github.com/coli-saar/am-parser/wiki/Training-the-Parser)).
Let's assume its files (`metrics.json`, `model.tar.gz` among others) should 
be written to some folder `MODELDIR` (folder will be created if it doesn't exist already), 
you have one GPU (cuda device 0),
and log output should be written to LOGFILE.
```bash
cd AMPARSERDIR
python -u train.py jsonnets/cogs2021/CONFIGFILE.jsonnet -s MODELDIR/  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  0  } }' &> LOGFILE
```

*Examples*  
On my local computer with the small debugging model:
```bash
# alternative: bash ./scripts/cogs2021/debugging_train.sh  
export LC_ALL=en_US.UTF-8  # I had problems with lr being implicitly set to 0 (in file: "lr": 0.01) with my de_DE.UTF-8 locale (in German , instead of . is used for floats)
python -u train.py jsonnets/cogs2021/debugging.jsonnet -s ~/HiwiAK/cogs2021/toy_model_run/training_output/  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  0  } }' &> ~/HiwiAK/cogs2021/toy_model_run/debugging_train.log
# tensorboard --logdir=../cogs2021/toy_model_run/training_output
```
Random example of a training call on the server (with comet.ml part stripped)
```bash
# note: with 'allennlp' conda environment, correct locale set, modified config file name, find a free GPU with nvidia-smi
time python3 -u train.py jsonnets/cogs2021/COGSallAutomaton_trainTokensMinVocab.jsonnet -s /local/piaw/cogs2021/first_experiments/auto3/models/prepreify/trainTokensMinVocab/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0  } }' &> /proj/irtg/sempardata/cogs2021/first_experiments/auto3prim/prepreify/training_trainTokensMinVocab.log
```


*Monitoring training progress*  
While training you can do `tensorboard --logdir=MODELDIR` to check model progress 
(important: last slash in path of MODELDIR should be omitted here).
Or you use [comet.ml](https://www.comet.ml/site/) with additional options (e.g. `--comet APIKEY --tags COGS auto3 train tokens --project cogs2021`) added to the `train.py` call.

*How long does training take?*  
Training time obviously depends on many factors such as 
- the number of epochs, batch size, size of the neural model and other hyperparameters
- the embedding type used (with BERT takes longer than with embeddings trained ffrom COGS only)
- the corpora used (full corpus? train or train100?)
- the hardware (GPU?)

Here are some examples: **to do: training times**
- local (pw): just minutes on a small train/dev (50 sentences train, 10 dev) for a small debugging model (32 word dim, 32/64 hidden, k=4 supertag decoding, batch size 1, no dropout) with early stopping (100 epochs, 20 patience).
- coli servers: about 6 to 10 hours, train100 takes longer than train, with BERT instead of token embeddings takes longer too.

**to do: maybe make a spreadsheet with results and link it here?**


### Step 3: (optional) predictions (using am-parser) and evaluation

#### (a) Get predictions using a trained model
Note that the training must be successfully completed (you should have a `model.tar.gz` in your `MODELDIR`).
Assume you want to store the output in some `OUTPUTDIR` (this folder must exist, but can be empty)
then the prediction will create 3 files in this folder:
- `COGS_gold.amconll` gold AM dependency trees
- `COGS_pred.amconll` predictions as AM dependency trees
- `COGS_pred.tsv` predictions in the native TSV format of COGS

If you are too lazy to call a separate official evaluation script with the generated tsv file,
note that the output of the bash prediction script (more concretely the `ToCOGSCorpus` programme called within this) will contain evaluation information too.

Run the following assuming that 
- you have one GPU available (`-g 0`) 
- you would like to save logs to `EVALLOGFILE`
- you would like to generate predictions for the `test.tsv` file found in `COGSDATADIR`

**With the fixed-tree decoder**  
```bash
cd AMPARSERDIR
bash ./scripts/cogs2021/unsupervised_predict.sh -i COGSDATADIR/test.tsv -o OUTPUTDIR -m MODELDIR/model.tar.gz -g 0 &> EVALLOGFILE
```
Note: you could add the `-f` option for fast (give up parameter set: when to back-off to using k-1 supertags)
- local (pw): `bash ./scripts/cogs2021/unsupervised_predict.sh -i ../cogs2021/small/test50.tsv -o ../cogs2021/toy_model_run/prediction_output_fixedt -m ../cogs2021/toy_model_run/training_output/model.tar.gz -g 0 -f &> ../cogs2021/toy_model_run/prediction_fixedt.log`
- coli severs, e.g. `bash ./scripts/cogs2021/unsupervised_predict.sh -i /proj/irtg/sempardata/cogs2021/data/COGS/data/test.tsv -o /proj/irtg/sempardata/cogs2021/first_experiments/auto3prim/predictions/test/repeat/train100BertLrA -m /local/piaw/cogs2021/first_experiments/auto3/models/repeat/train100BertLrA_repeat1/model.tar.gz -g 0 -f &> /proj/irtg/sempardata/cogs2021/first_experiments/auto3prim/predictions/test/repeat/repeat1_test_train100BertLrA.log`
(see also [the am-parser wiki on prediction and evaluation on test data](https://github.com/coli-saar/am-parser/wiki/Prediction-and-evaluation-on-test-data),
but due to using the 'unsupervised' approach and a new formalism COGS might not be applicable).

**With the projective/Astar decoder** 
You have to add the `-p` option to the bash script.
Also have to use a second am-tools jar file **to do: which am-tools version**?
```bash
cd AMPARSERDIR
bash ./scripts/cogs2021/unsupervised_predict.sh -i ../cogs2021/small/test50.tsv -o ../cogs2021/toy_model_run/prediction_output_astar -m ../cogs2021/toy_model_run/training_output/model.tar.gz -g 0 -p &> ../cogs2021/toy_model_run/prediction_astar.log
```
e.g. on the coli server
```bash
cd /local/piaw/am-parser
time bash ./scripts/cogs2021/unsupervised_predict.sh -i /proj/irtg/sempardata/cogs2021/data/COGS/data/gen500.tsv -o /proj/irtg/sempardata/cogs2021/first_experiments/auto3prim/predictions/gen/astar/trainTokenLrA -m /local/piaw/cogs2021/first_experiments/auto3/models/trainTokenLrA/model.tar.gz -g 1 -p &> /proj/irtg/sempardata/cogs2021/first_experiments/auto3prim/predictions/gen/astar/firsttry_trainTokenLrAVocab.log
```

#### (b) Computing evaluation metrics
The COGS authors use exact match accuracy as the main evaluation metric and
reported overall results as well as result per generalization type (see 3rd column in the TSV).
In their appendix they also mention token-level edit distance and ill-formedness (closing parenthesis missing).
As of now (May 2021) there doesn't seem to be a separate evaluation script.
(first author told me a while ago they computed their accuracy values using OpenNMT which they also used to train their models).
**to do: which eval script used? incl generalization type specific results?**

As mentioned in the last section, the call of the prediction script will compute metrics too,
but if you don't trust this computation, 
you can use the the `COGS_pred.tsv` file in the output directory of the prediction script as input to whatever evaluation script you like.
The format of the tsv file is the same as the gold files.

## experiments

### jsonnet files
in `jsonnets/cogs2021/`

Toy model debugging on local computer: `debugging.jsonnet`
Example configuration for a non-toy model: `COGSallAutomaton.jsonnet`, mostly based on `jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet`

You might be interested in tweaking different factors, for instance:
- training set (`train` or `train100`), relevant for the `train_zip_corpus_path` for instance
- embeddings: from BERT (`bert`), learnt from cogs alone (`tokens`)
- number of sources: 3,4,5?? (that's implicitly done by selecting input zip data)
- supervised loss for edge existence and lexical labels: `"all_automaton_loss": BOOLEAN`, setting it to true means all loss flows through automata, false means supervised loss for edge existence and lex label.
- edge model: `kg_edges` or `kg_rel_edges` (relative distance encodings)

Jsonnet files of the format `COGS_{train100,train}{Bert,Token}_{kg,kgrel}.jsonnet`
- all have in common:
  - supervised loss for edge existence and lex label scores and 60 epochs correspondingly
  - lr reduced (0.0001) compared to AMR
  - no early stopping but final parameters selected based on dev set exact match
- `train` refers to `train.tsv` content as input, `train100`to `train_100.tsv` training data (see `train_zip_corpus_path` in jsonnet file)
- `Bert` refers to using Bert embeddings, `Token` to using embeddings learnt from COGS data alone (see `embedding_name` in jsonnet file)
- `kg` refers to using the KG dependency edge model, `kgrel` refers to using the KG model enhanced with relative distance encodings for edge existence scores
  (implementation: `graph_dependency_parser/components/edge_models/` KG or KGrel)


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

Note: currently, all this is based on the 'unsupervised' parser of [Groschwitz et al. 2021](https://aclanthology.org/2021.spnlp-1.3)
(see also the respective branches in am-parser and am-tools): that is, source names are learnt!


# References and links

- The COGS dataset: Kim and Linzen (2020)
  - paper: Kim and Linzen (2020)
  [COGS: A Compositional Generalization Challenge Based on Semantic Interpretation](https://www.aclweb.org/anthology/2020.emnlp-main.731/).
  In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*
  Online. Association for Computational Linguistics. pp. 9087-9105. doi: 10.18653/v1/2020.emnlp-main.731
  - [github: COGS dataset and their models](https://github.com/najoungkim/COGS)
of [Kim & Linzen (2020)](https://www.aclweb.org/anthology/2020.emnlp-main.731/).
- The 'unsupervised parser' (for which we added COGS as a new formalism)
  - paper: Groschwitz, Fowlie and Koller (2021): 
    [Learning compositional structures for semantic graph parsing](https://aclanthology.org/2021.spnlp-1.3). 
    In *Proceedings of the 5th Workshop on Structured Prediction for NLP (SPNLP 2021)*.
    Online. Association for Computational Linguistics. pp. 22-36. doi: 10.18653/v1/2021.spnlp-1.3
  - code: [`unsupervised2020` branch of `am-parser`](https://github.com/coli-saar/am-parser/tree/unsupervised2020)  (see specifically the markdown [files under `graph_dependency_parser/inside_maximization`](https://github.com/coli-saar/am-parser/tree/unsupervised2020/graph_dependency_parser/inside_maximization))
- branches with COGS additions:
  - [`cogs_unsupervised` branch of the `am-parser` repo](https://github.com/coli-saar/am-parser/tree/cogs_unsupervised)
  - [`cogs_new_decomp` branch of the `am-tools` repo](https://github.com/coli-saar/am-tools/tree/cogs_new_decomp)
- the [AM parser wiki](https://github.com/coli-saar/am-parser/wiki)
- about different decoders
  - fixed-tree decoder is described in [Groschwitz et al. 2018](https://aclanthology.org/P18-1170/), section 6.1, p.6
  - projective Astar decoder is described in [Lindemann et al. 2020](https://aclanthology.org/2020.emnlp-main.323/), section 4, p.4


# PWs notes on local debugging

These are my notes for debugging a small toy model on my local computer.

## Gotchas

- don't forget to copy most recent am-tools.jar into am-parser directory!
- same for the second `master-am-tools.jar` if you use the Astar decoder!
- don't forget to change the file paths in config files and bash scripts 
(am-tools train-dev prepare, jsonnet files, command line arguments for train and predict)
- on my local computer hat to switch to `LC_ALL=en_US.UTF-8`, otherwise `0.01` was interpreted as `0`(German locale uses `,` instead of `.` for floats)
- also if you re-run the pipeline, make sure that errors are not hidden by accidentally using a file created in the last run. Check the log files for error messages.
- options to play around with:
  - `get_train_dev.sh`: 
    - if you would like to enable *preposition reification*, add the `-r` option. Otherwise prepositions are only represented as edges.
    - how many sources? 3 are needed at least to deal with ditransitive verbs ("Ava gave Ben the cookie"), but more are possible too.
  - training (jsonnet file):
    - not for toy model, but can experiment with dropout
    - in general: number of layers, layer dimensionality, embeddings, edge models, loss mixing, `all_automaton` true or not?
  - `unsupervised_predict.sh`: you can choose between two decoders
    - the fixed-tree decoder (described in [Groschwitz et al. 2018](https://aclanthology.org/P18-1170/), section 6.1, p.6). That's the default. It is recommended to add the `-f` option for faster decoding (give up time limit set)
    - the *projective Astar decoder* (described in [Lindemann et al. 2020](https://aclanthology.org/2020.emnlp-main.323/), section 4, p.4). You can enable it by using the `-p` option. Important note: you might want to use an additional am-tools jar file build from the master branch when calling the Astar in `unsupervised_predict`: fixed punctuation-not-ignored bug.


## Commands used

Toy model: as described in `jsonnets/cogs2021/debugging.jsonnet`
Toy data: first 50 samples from train.tsv, first 10 from dev.tsv, first 50 from test.tsv

Preparing training input is quick, 
Training toy model takes less than 10 minutes, 
but results are not really good (0 exact match: label accuracy: is it OOV words in the dev set?).
Prediction takes less than a minute for 50 test sentences.
```bash
cd ~/HiwiAK/am-parser/
# (1) get train and dev data. Option: add -r to use preposition reification, add -e TESTFILE for test.amconll s.t evaluate_on_test : true  works in the jsonnet file
bash ./scripts/cogs2021/get_train_dev.sh -t ~/HiwiAK/cogs2021/small/train50.tsv -d ~/HiwiAK/cogs2021/small/dev10.tsv -o ~/HiwiAK/cogs2021/toy_model_run/training_input/ -e ~/HiwiAK/cogs2021/small/test50.tsv -r -s 3 -p dp_dev &> ../cogs2021/toy_model_run/get_train_dev.log
# (2) train
# alternative: bash ./scripts/cogs2021/debugging_train.sh  
export LC_ALL=en_US.UTF-8  # I had problems with lr being set to 0 (in file: "lr": 0.01) with my de_DE.UTF-8
python -u train.py jsonnets/cogs2021/debugging.jsonnet -s ~/HiwiAK/cogs2021/toy_model_run/training_output/  -f --file-friendly-logging  -o ' {"trainer" : {"cuda_device" :  0  } }' &> ~/HiwiAK/cogs2021/toy_model_run/debugging_train.log
# tensorboard --logdir=../cogs2021/toy_model_run/training_output
# (3) predict. Option: choose projective Astar decoder by using -p
bash ./scripts/cogs2021/unsupervised_predict.sh -i ../cogs2021/small/test50.tsv -o ../cogs2021/toy_model_run/prediction_output_fixedt -m ../cogs2021/toy_model_run/training_output/model.tar.gz -g 0 -f &> ../cogs2021/toy_model_run/prediction_fixedt.log
bash ./scripts/cogs2021/unsupervised_predict.sh -i ../cogs2021/small/test50.tsv -o ../cogs2021/toy_model_run/prediction_output_astar -m ../cogs2021/toy_model_run/training_output/model.tar.gz -g 0 -p &> ../cogs2021/toy_model_run/prediction_astar.log
## bash ./scripts/predict.sh -i ../cogs2021/small/test50.tsv -T COGS -o ../cogs2021/toy_model_run/prediction_output -m ../cogs2021/toy_model_run/training_output_fixedt/model.tar.gz -g 0 -f &> ../cogs2021/toy_model_run/prediction_fixedt.log
```


**Astar parser/projective decoder**  
For the Astar parser we need a separate second am-tools jar file built from the [master branch](https://github.com/coli-saar/am-tools/tree/master) instead of the [cogs one](https://github.com/coli-saar/am-tools/tree/cogs_new_decomp).
This jar file will be called `master-am-tools.jar` to not confuse it with the other (`am-tools.jar` built from the cogs branch).  
Make sure this second jar file is used when the Astar decoder is called! (prediction script contains `use_second_jar=true` and valid path to this file in `astar_jar` variable.)
Important note (July 8): bug fix of off-by-one error resulting in wrong prediction for final word (punctuation mark couldn't be ignored): only on master!.  
Second version: (ultimately added `-p` option to `unsupervised_predict.sh`, still model was _trained_ with fixed-tree decoder for validation?)
```bash
bash ./scripts/cogs2021/unsupervised_predict.sh -i ../cogs2021/small/test50.tsv -o ../cogs2021/toy_model_run/prediction_output_astar -m ../cogs2021/toy_model_run/training_output/model.tar.gz -g 0 -p &> ../cogs2021/toy_model_run/prediction_astar.log
```

**experiment with preposition reification**  
flag to reify prepositions: (in java: `--reifyprep`): need to add `-r` flag to `get_train_dev.sh`-call
```bash
cd ~/HiwiAK/am-parser/
bash ./scripts/cogs2021/get_train_dev.sh -t ../cogs2021/small/train50.tsv -d ../cogs2021/small/dev10.tsv -o ../cogs2021/toy_model_run/training_input/ -s 3 -p dp_dev -r
```

## Environment used locally

On the coli servers, used the `allennlp` conda environment (cf. the AM parser wiki).
On my local computer:
Conda environment
```yaml
channels:
  - defaults
dependencies:
  - numpy=1.16.4
  - pip=19.1.1
  - python=3.7.3
  - setuptools=41.0.1
  - wheel=0.33.4
  - pip:
    - allennlp==0.8.4
    - comet-ml==3.1.6
    - conllu==0.11
    - cython==0.29.7
    - graphviz==0.14.1
    - nltk==3.4.1
    - overrides==1.9
    - penman==0.12.0
    - pyjnius==1.2.1
    - scikit-learn==0.20.3
    - scipy==1.2.1
    - spacy==2.1.8
    - tensorboard==1.13.1
    - tensorboardx==1.6
    - tensorflow==1.13.1
    - torch==1.1.0
```
and a few additional installations necessary:
```bash
pip install git+https://github.com/andersjo/dependency_decoding
python -m spacy download en_core_web_md
```
