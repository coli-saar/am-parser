# am-parser
Modular implementation of the AM dependency parser used in [Compositional Semantic Parsing Across Graphbanks](https://www.aclweb.org/anthology/P19-1450).

### Running the code
The usage of the code is documented in the [wiki pages](https://github.com/coli-saar/am-parser/wiki/Train-Parser).
__We will update the documentation soon to make it clearer and easier to reproduce our experiments.__

### Requirements
- Python 3.7
- AllenNLP (tested with version 0.8.4 and Pytorch 1.1)
- Cython
- [am-tools](https://github.com/coli-saar/am-tools) to prepare the training data and to evaluate the AM dependency trees to graphs
- [dependency_decoding](https://github.com/andersjo/dependency_decoding)

It is best, to set up a conda environment for experimentation with the parser.

__Internal note:__ this is already set up on the Saarland servers, see details [here](https://github.com/coli-saar/am-parser/wiki/Setup-on-the-Saarland-servers).

### Things to play around with
The configuration files have many places where you can make changes and see how it affects parsing performance.
There are currently two edge models implemented, the Dozat & Manning 2016 and the Kiperwasser & Goldberg 2016 one.
Apart from the edge models, the are also two different loss functions, a softmax log-likelihood and a hinge loss that requires running the CLE algorithm at training time.

