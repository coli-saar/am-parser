rm -rf models/stupid1/* #clean up

allennlp train non-elmo_config.jsonnet -s models/stupid1 --include-package graph_dependency_parser

