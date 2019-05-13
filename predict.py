
import graph_dependency_parser.graph_dependency_parser #important import
from graph_dependency_parser.components.evaluation.predictors import ConlluPredictor


config_file = 'non-elmo_config.jsonnet'
serialization_dir = "models/stupid1/"
input_file = "data/en_ewt-ud-dev.conllu"
output_file = "output.conllu"

predictor = ConlluPredictor.from_config(config_file,serialization_dir)
predictor.parse_and_save(input_file,output_file)
