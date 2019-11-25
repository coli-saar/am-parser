# am-parser
Modular implementation of the AM dependency parser used in [Compositional Semantic Parsing Across Graphbanks](https://www.aclweb.org/anthology/P19-1450).

## Quick Guide
This is a quick guide on how to use our already trained models to make predictions, either for official test data to reproduce our results, or on arbitrary sentences.

You can find documentation on how to train the parser in the [wiki pages](https://github.com/coli-saar/am-parser/wiki/Train-Parser) (we will update this part of the documentation later in fall 2019 to make it easier for you to train your own models).

### Requirements
- Python 3.7 up to version 3.7.3
- Python 2.7 for EDS and AMR evaluation (EDM metric and Smatch)
- AllenNLP (tested with version 0.8.4 and Pytorch 1.1)
- Cython
- [dependency_decoding](https://github.com/andersjo/dependency_decoding)

(We recommend to set up a conda environment.)

__Internal note:__ this is already set up on the Saarland servers, see details [here](https://github.com/coli-saar/am-parser/wiki/Setup-and-file-locations-on-the-Saarland-servers).

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

### Getting graphs from raw text
From the main directory, run `bash scripts/predict_from_raw_text.sh` with the following arguments (or with -h for help):
* `-i` the input file with one sentence per line. These must already be tokenized. An example is in `example/input.txt`.
* `-T` the type of graph bank you want to parse for, options are DM, PAS, PSD, EDS or AMR.
* `-o` the desired output folder (this will contain the final parsing output, but also several intermediary files)

For example, say you want to do DM parsing and `INPUT` is the path to your sdp file, then
```
bash scripts/predict_from_raw_text.sh -i example/input.txt -T DM -o example/
``` 
will create a file `DM.sdp` in the `example` folder with graphs for the sentences in `example/input.txt`.

### Notes

* When you run either of the above commands for the first time, they will download large files: our trained model file, a compiled `jar` file to support output graph formats, as well as BERT embeddings.
* This uses the BERT multitask version. In particular, the AMR 2017 training set was used and results on the AMR 2015 test set are not comparable.

## Things to play around with
When training your own model, the configuration files have many places where you can make changes and see how it affects parsing performance.
There are currently two edge models implemented, the Dozat & Manning 2016 and the Kiperwasser & Goldberg 2016 one.
Apart from the edge models, the are also two different loss functions, a softmax log-likelihood and a hinge loss that requires running the CLE algorithm at training time.

