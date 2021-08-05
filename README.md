# am-parser
The AM parser is a compositional semantic parser with high accuracy across a wide range of graphbanks. This repository is a modular implementation based on AllenNLP and Pytorch.

## Online Demo
Try out the [online demo](http://amparser.coli.uni-saarland.de:8080/) of our parser!

## Papers
This repository (jointly with its sister-repository at [am-tools](github.com/coli-saar/am-tools)) contains the code for several papers:

- [Compositional Semantic Parsing Across Graphbanks](https://www.aclweb.org/anthology/P19-1450) (Lindemann et al. 2019) demonstrates the parsers accuracy across graphbanks. Most of the documentation in this readme and in the [wiki](https://github.com/coli-saar/am-parser/wiki) refers to this paper.
- [Learning compositional structures for semantic graph parsing](https://aclanthology.org/2021.spnlp-1.3/) (Groschwitz et al. 2021) describes a method to learn the AM dependency trees as latent compositional structures. Its code documentation is currently being built in the wiki [here](https://github.com/coli-saar/am-parser/wiki/Learning-compositional-structures).
- [Fast semantic parsing with well-typedness guarantees](https://aclanthology.org/2020.emnlp-main.323/) (Lindemann et al. 2020) presents AM parsing methods with faster inference. The top-down parser resides [in its own repo](https://github.com/coli-saar/am-transition-parser), but the A* parser lives here (see its [wiki page](https://github.com/coli-saar/am-parser/wiki/A*-Parser)).
- [Normalizing Compositional Structures Across Graphbanks](https://aclanthology.org/2020.coling-main.267/) (Donatelli et al. 2020) discusses how compositional structures can be different across graphbanks, and how they can be normalized.
- [Saarland at MRP 2019: Compositional parsing across all graphbanks](https://aclanthology.org/K19-2006/) describes our participation at the MRP 2019 shared task.
- [AMR dependency parsing with a typed semantic algebra](https://aclanthology.org/P18-1170/) (Groschwitz et al. 2018) used an older, now deprecated codebase but describes the parser in most detail, and the general principles still apply in this repository.

## Setup

### Requirements
- Python 3.7 up to version 3.7.3
- Python 2.7 for EDS and AMR evaluation (EDM metric and Smatch)
- AllenNLP (tested with version 0.8.4 and Pytorch 1.1)
- Cython
- comet-ml
- overrides (tested with versions 1.9 and 3.1.0; newer versions lead to errors, see https://github.com/allenai/allennlp/issues/5197)
- [dependency_decoding](https://github.com/andersjo/dependency_decoding)
- The spacy core web md model: `python -m spacy download en_core_web_md`
- You may require to set your version of sklearn (an AllenNLP requirement, usually automatically installed) manually to version 0.22 or lower, e.g. with `pip install scikit-learn==0.22.2`
- Java (tested with Java 11)
- a build of [am-tools](https://github.com/coli-saar/am-tools); will be downloaded automatically.

(We recommend to set up a conda environment.)
**If you still have problems running the parser** check the list of [third party packages](https://github.com/coli-saar/am-parser/wiki/Third-Party-Packages) in the Wiki. This list also contains packages necessary to run branches other than the Master branch.

__Internal note:__ this is already set up on the Saarland servers, see details [here](https://github.com/coli-saar/am-parser/wiki/Setup-and-file-locations-on-the-Saarland-servers).

## Quick Guide to the Pretrained Models of Lindemann et al. (2019)
This is a quick guide on how to use our already trained models to make predictions, either for official test data to reproduce our results, or on arbitrary sentences.

**You can find documentation on how to train the parser on the [wiki pages](https://github.com/coli-saar/am-parser/wiki).**


### Reproducing our experiment results

From the main directory, run `bash scripts/predict.sh` with the following arguments (or with -h for help):
* `-i` the input file, e.g. for the SDP corpora (DM, PAS, PSD) a `.sdp` file such as the `en.id.dm.sdp` in-domain test file of the DM corpus. For EDS, make this the test.amr file that contains the gold graphs in PENMAN notation. For AMR, use the directory which contains all the test corpus files (e.g. data/amrs/split/test/ in the official AMR corpora). You must provide these files.
* `-T` the type of graph bank you want to parse for, the options are DM, PAS, PSD, EDS or AMR
* `-o` the desired output folder (this will contain the final parsing output, but also several intermediary files)

For example, say you want to do DM parsing and `INPUT` is the path to your sdp file, then
```
bash scripts/predict.sh -i INPUT -T DM -o example/
``` 
will create a file `DM.sdp` in the `example` folder with graphs for the sentences in `INPUT`, as well as print evaluation scores compared to the gold graphs in `INPUT`.

With this pre-trained model (this is the MTL+BERT version, corresponding to the bottom-most line in Table 1 in the paper) you should get (labeled) F-scores close to the following on the test sets:

| DM id | DM ood | PAS id| PAS ood| PSD id | PSD ood | EDS (Smatch) | EDS (EDM) | AMR 2017 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 94.1 | 90.5 | 94.9 | 92.9 | 81.8 | 81.6 | 90.4  | 85.2 | 76.3 |

The F-score for AMR 2017 is considerably better than published in the paper and stems from fixing bugs in the postprocessing.
Please note that these evaluation scores were obtained __without__ the `-f` option 
and your results might differ slightly depending on your CPU because the parser uses a timeout. This is mainly relevant for AMR. We used Intel Xeon E5-2687W v3 processors.


### Getting graphs from raw text
From the main directory, run `bash scripts/predict_from_raw_text.sh` with the following arguments (or with -h for help):
* `-i` the input file with one sentence per line. These must already be tokenized. An example is in `example/input.txt`.
* `-T` the type of graph bank you want to parse for, options are DM, PAS, PSD, EDS or AMR.
* `-o` the desired output folder (this will contain the final parsing output, but also several intermediary files)

For example, say you want to do DM parsing and make predictions for the sentences in `example/input.txt`, then
```
bash scripts/predict_from_raw_text.sh -i example/input.txt -T DM -o example/
``` 
will create a file `DM.sdp` in the `example` folder with graphs for the sentences in `example/input.txt`.

### Notes

* When you run either of the above commands for the first time, they will download large files: our trained model file, a compiled `jar` file to support output graph formats, as well as BERT embeddings.
* This uses the BERT multitask version. In particular, the AMR 2017 training set was used and results on the AMR 2015 test set are not comparable. 
* When parsing graphs from raw text, the model used was trained without embeddings for lemmas, POS tags and named entities and thus is __not directly comparable to the results from the paper__.
* In contrast to the ACL 2019 experiments, we now use a [new formalization of the type system](http://www.coli.uni-saarland.de/~jonasg/thesis.pdf). 
If you absolutely want to use the old implementation and formalization, use the `old_types` branch and a version of am-tools from February 2020.

After the bugix in AMR postprocessing, the parser achieves the following Smatch scores on the test set (average of 5 runs and standard deviations):

|  | AMR 2015 | AMR 2017 |
| --- | --- | --- |
| Single task, GloVe | 70.0 +- 0.1 | 71.2 +- 0.1 |
| Single task, BERT | 75.1 +- 0.1 | 76.0 +- 0.2 |

## Things to play around with
When training your own model, the configuration files have many places where you can make changes and see how it affects parsing performance.
There are currently two edge models implemented, the Dozat & Manning 2016 and the Kiperwasser & Goldberg 2016 one.
Apart from the edge models, the are also two different loss functions, a softmax log-likelihood and a hinge loss that requires running the CLE algorithm at training time.


## Third Party Packages

An overview of the third party packages we used can be found at [https://github.com/coli-saar/am-parser/wiki/Third-Party-Packages](https://github.com/coli-saar/am-parser/wiki/Third-Party-Packages).
