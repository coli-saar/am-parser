local num_epochs = 30;
local device = 2;
local pos_dim = 32;
local lemma_dim = 64;
local ner_dim = 16;

local elmo_path = "/local/mlinde/elmo/";
local encoder_output_dim = 512;
local final_encoder_output_dim = 2 * encoder_output_dim;

#local train_data = "data/EDS/train/train.amconll";
local train_data = "data/AMR/2015/train/train.amconll";
local dev_data = "data/AMR/2015/gold-dev/gold-dev.amconll";
#local dev_data = "data/EDS/gold-dev/gold-dev.amconll";


local dataset_reader = {
        "type": "amconll",
        "token_indexers": {
            "elmo": {
              "type": "elmo_characters"
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
        "type": "pure_supertagger",
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
                "dropout" : [0.5],
                "activations" : "tanh"
            }

        },
        "lexlabeltagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [1024],
                "dropout" : [0.5],
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
            "token_embedders": {
                "elmo" : {
                    "type": "elmo_token_embedder",
                        "options_file": elmo_path+"elmo_2x4096_512_2048cnn_2xhighway_options.json",
                        "weight_file": elmo_path+"elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                        "do_layer_norm": false,
                        "dropout": 0.0
                    },
             }
        },

        #LOSS:
        "loss_mixing" : {
            "supertagging": 1.0,
            "lexlabel": 1.0
        },

        "supertagger_loss" : { }, #for now use defaults
        "lexlabel_loss" : { },

    },
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    "trainer": {
        "num_epochs": num_epochs,
         "cuda_device": device,
        "optimizer": {
            "type": "adam",
        },
        "validation_metric" : "+Supertagging_Acc",
        "num_serialized_models_to_keep" : 1
    }
}

