local num_epochs = 30;
local device = 1;
local pos_dim = 32;
local lemma_dim = 64;
local ner_dim = 16;


local eval_commands = import 'eval_commands.libsonnet';

local encoder_output_dim = 512;
local final_encoder_output_dim = 2 * encoder_output_dim;

#local train_data = "data/SemEval/2015/DM/train/train.amconll";
#local train_data = "data/SemEval/2015/DM/gold-dev/gold-dev.amconll";
local train_data = "data/AMR/2015/train/train.amconll";
local dev_data = "data/AMR/2015/gold-dev/gold-dev.amconll";
#local dev_data = "data/SemEval/2015/DM/gold-dev/gold-dev.amconll";

local bert_model = "bert-large-uncased";

local dataset_reader = {
        "type": "amconll",
        "token_indexers": {
            "bert": {
              "type": "bert-pretrained",
              "pretrained_model": bert_model,
            }
        }
      };
local data_iterator = {
        "type": "bucket",
        "batch_size": 64,
        "sorting_keys": [
            [
                "words",
                "num_tokens"
            ]
        ]
    };

{
    "dataset_reader": dataset_reader,
    "iterator": data_iterator,
     "vocabulary" : {
            "min_count" : {
            "lemmas" : 7
                }
     },
    "model": {
        "type": "graph_dependency_parser",
        "edge_model" : {
            "type" : "kg_edges",
            "encoder_dim" : final_encoder_output_dim,
            "label_dim": 300,
            "edge_dim": 300,
            #"activation" : "tanh",
            "dropout": 0.0,
        },
        "dropout": 0.3,
        "input_dropout": 0.3,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "num_layers": 2,
            "recurrent_dropout_probability": 0.4,
            "layer_dropout_probability": 0.2,
            "use_highway": false,
            "hidden_size": encoder_output_dim,
            "input_size": 1024 + pos_dim + lemma_dim + ner_dim
        },
        "supertagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [1024],
                "dropout" : [0.4],
                "activations" : "tanh"
            }

        },
        "lexlabeltagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [1024],
                "dropout" : [0.4],
                "activations" : "tanh"
            }

        },
        "pos_tag_embedding":  {
           "embedding_dim": pos_dim,
           "vocab_namespace": "pos"
        },
        "lemma_embedding":  {
           "embedding_dim": lemma_dim,
           "vocab_namespace": "lemmas"
        },
         "ne_embedding":  {
           "embedding_dim": ner_dim,
           "vocab_namespace": "ner_labels"
        },

        "text_field_embedder": {
            "type": "basic",
            "allow_unmatched_keys" : true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"] },
            "token_embedders": {
                "bert" : {
                    "type": "bert-pretrained",
                        "pretrained_model" : bert_model,
                    },
             }
        },

        #LOSS:
        "loss_mixing" : {
            "edge_existence" : 1.0,
            "edge_label": 1.0,
            "supertagging": 1.0,
            "lexlabel": 1.0
        },
        "loss_function" : {
            "existence_loss" : { "type" : "kg_edge_loss", "normalize_wrt_seq_len": false},
            "label_loss" : {"type" : "dm_label_loss" , "normalize_wrt_seq_len": false}
        },

        "supertagger_loss" : { }, #for now use defaults
        "lexlabel_loss" : { },

          #optional: set validation evaluator that is called after each epoch.
        "validation_evaluator": {
            "system_input" : "data/AMR/2015/dev/dev.amconll",
            "gold_file": "dev",
            "use_from_epoch" : 10,
            "predictor" : {
                    "type" : "amconll_predictor",
                    "dataset_reader" : dataset_reader, #same dataset_reader as above.
                    "data_iterator" : data_iterator, #same bucket iterator also for validation.
                    "k" : 6, #k-best supertags
                    "threads" : 4,
                    "give_up": 10.0, #try parsing only for 1 second, then retry with smaller k
                    "evaluation_command" : eval_commands['AMR-2015']
            }
        }

    },
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    "trainer": {
        "num_epochs": num_epochs,
         "cuda_device": device,
        "optimizer": {
            "type": "adam",
        },
        "validation_metric" : "+LAS",
        "num_serialized_models_to_keep" : 1
    }
}

