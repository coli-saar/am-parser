import tempfile

import graph_dependency_parser #important import

from allennlp.commands.train import train_model

from allennlp.common.params import Params


params = Params.from_file('non-elmo_config.jsonnet')
with tempfile.TemporaryDirectory() as serialization_dir:
    model = train_model(params, serialization_dir)
