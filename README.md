# graph-dependency-parser
Modular implementation of a graph-based first-order dependency parser in AllenNLP. 
Decoding is performed with the Chu-Liu/Edmonds (CLE) algorithm.

There are currently two edge models implemented, the Dozat & Manning 2016 and the Kiperwasser & Goldberg 2016 one.
Apart from the edge models, the are also two different loss functions, a softmax log-likelihood and a hinge loss that requires running the CLE algorithm at training time.

### Requirements
- Python 3.7
- AllenNLP (tested with version 0.8.4 and Pytorch 1.1)
- Cython
- [dependency_decoding](https://github.com/andersjo/dependency_decoding)

It is best, to set up a conda environment.

__Internal note: use /proj/irtg.shadow/conda/envs/allennlp__

### Running the code
Two example configurations are provided (.jsonnet files). To see if everything's working and to train a parser (on the EWT corpus), run:

```bash
mkdir -p data/
cd data/
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-train.conllu
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-dev.conllu

mkdir -p models/
bash example_train.sh

echo Evaluating on development set

bash example_evaluate.sh
```

If you play around with the code in an IDE you might want to use `run.py` instead of the bash scripts.
