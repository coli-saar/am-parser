# A small example config.

local num_epochs = 130; # small training corpus requires lots of epochs.
local device = 0;

local word_dim = 128;
local char_dim = 16;
local num_filters = 50;
local filters = [3];
local max_filter = 3; //KEEP IN SYNC WITH filters!

local pos_embedding = 32;

local encoder_dim = 256;


local dropout_in = 0.33;

#local eval_commands = import "eval_commands.libsonnet";

local additional_lexicon = {
     "sublexica" : {
            "edge_labels" : "data/example_DM/lexicon/edges.txt",
            "constants" : "data/example_DM/lexicon/constants.txt",
            "term_types" : "data/example_DM/lexicon/types.txt",
            "lex_labels" : "data/example_DM/lexicon/lex_labels.txt"
     }
} ;

local transition_system = {
    "type" : "ltl", #choices are: "dfs-children-first", "ltl", "ltf", or "dfs"
    "children_order" : "IO",
    "pop_with_0" : true,
    "additional_lexicon" : additional_lexicon,
};

local dataset_reader = {
               "type": "amconll",
               "transition_system" : transition_system,
               "workers" : 1,
               "overwrite_formalism" : "amr",
               "run_oracle" : true,
               "fuzz" : true,
               "token_indexers" : {
                    "tokens" : {
                        "type": "single_id",
                         "lowercase_tokens": true
                               },
                    "token_characters" : {
                        "type" : "characters",
                        "min_padding_length" : max_filter
                    }
               }


           };

local data_iterator = {
        "type": "same_formalism",
        "batch_size": 16,
       "formalisms" : ["amr"]
    };


{
    "dataset_reader": dataset_reader,
    "validation_dataset_reader" : dataset_reader,

    "validation_command" : {

        "type" : "bash_evaluation_command",
        "command" : "python3 topdown_parser/evaluation/am_dep_las.py {gold_file} {system_output}",

        "result_regexes" : {
            "Constant_Acc" : [4, "Supertagging acc % (?P<value>[0-9.]+)"],
            "Lex_Acc" : [5, "Lexical label acc % (?P<value>[0-9.]+)"],
            "UAS" : [6, "UAS.* % (?P<value>[0-9.]+)"],
            "LAS" : [7, "LAS.* % (?P<value>[0-9.]+)"],
            "Content_recall" : [8, "Content recall % (?P<value>[0-9.]+)"]
        }
    },



    "iterator": data_iterator,
    "model": {
        "type": "topdown",
        "transition_system" : transition_system,

        "input_dropout" : dropout_in,
        "encoder_output_dropout" : 0.2,

        "context_provider" : {
            "type" : "sum",
            "providers" : [
                  {"type" : "most-recent-child" }
            ]
        },


        "supertagger" : {
            "type" : "combined-tagger",
            "lexicon" : additional_lexicon,
            "namespace" : "constants",
            "mlp" : {
                "input_dim" : 2*2*encoder_dim,
                "num_layers" : 1,
                "hidden_dims" : 1024,
                "dropout" : 0.0,
                "activations" : "tanh",
            }
        },

        "lex_label_tagger" : {
            "type" : "combined-tagger",
            "lexicon" : additional_lexicon,
            "namespace" : "lex_labels",
            "mlp" : {
                "input_dim" : 2*2*encoder_dim,
                "num_layers" : 1,
                "hidden_dims" : 1024,
                "dropout" : 0.0,
                "activations" : "tanh",
            }
        },

// Comment this in if you use LTF:
//        "term_type_tagger" : {
//            "type" : "combined-tagger",
//            "lexicon" : additional_lexicon,
//            "namespace" : "term_types",
//            "mlp" : {
//                "input_dim" : 2*2*encoder_dim,
//                "num_layers" : 1,
//                "hidden_dims" : 1024,
//                "dropout" : 0.0,
//                "activations" : "tanh",
//            }
//        },

        "encoder" : {
            "type" : "lstm",
            "input_size" :  num_filters + word_dim + pos_embedding,
            "hidden_size" : encoder_dim,
            "bidirectional" : true,
        },


        "tagger_encoder" : {
            "type" : "lstm",
            "input_size" :  num_filters + word_dim + pos_embedding,
            "hidden_size" : encoder_dim,
            "bidirectional" : true,
        },

        "decoder" : {
            "type" : "ma-lstm",
            "input_dim": 2*encoder_dim,
            "hidden_dim" : 2*encoder_dim,
            "input_dropout" : 0.2,
            "recurrent_dropout" : 0.1
        },
        "text_field_embedder": {
               "tokens": {
                    "type": "embedding",
                    "embedding_dim": word_dim
                },
            "token_characters": {
              "type": "character_encoding",
                  "embedding": {
                    "embedding_dim": char_dim
                  },
                  "encoder": {
                    "type": "cnn",
                    "embedding_dim": char_dim,
                    "num_filters": num_filters,
                    "ngram_filter_sizes": filters
                  },
              "dropout": dropout_in
            }
        },
        "edge_model" : {
            "type" : "mlp",
            "encoder_dim" : 2*encoder_dim,
            "hidden_dim" : 256,
        },
        "edge_label_model" : {
            "type" : "simple",
            "lexicon" : additional_lexicon,
            "mlp" : {
                "input_dim" : 2*2*encoder_dim,
                "num_layers" : 1,
                "hidden_dims" : [256],
                "activations" : "tanh",
                "dropout" : 0.2
            }
        },
        "edge_loss" : {
            "type" : "nll"
        },

        "pos_tag_embedding" : {
            "embedding_dim" : pos_embedding,
            "vocab_namespace" : "pos"
        }

    },
    "train_data_path": "data/example_DM/train/train.amconll",
    "validation_data_path": "data/example_DM/gold-dev/gold-dev.amconll",

    "evaluate_on_test" : false,

    "trainer": {
        "num_epochs": num_epochs,
        "cuda_device": device,
        "optimizer": {
            "type": "adam",
            "betas" : [0.9, 0.9]
        },
        "num_serialized_models_to_keep" : 1,
        "epochs_before_validate" : 2,
        "validation_metric" : "+LAS"
    },

    "dataset_writer":{
      "type":"amconll_writer"
    },

    "annotator" : {
        "dataset_reader": dataset_reader,
        "data_iterator": data_iterator,
        "dataset_writer":{
              "type":"amconll_writer"
        }
    },

}

