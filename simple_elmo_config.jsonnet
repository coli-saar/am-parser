local num_epochs = 10;
local device = 1;
local pos_dim = 50;
local lemma_dim = 50;
local ner_dim = 32;

local elmo_path = "/local/mlinde/elmo/";
local encoder_output_dim = 512;
local final_encoder_output_dim = 2 * encoder_output_dim;

#local train_data = "data/SemEval/2015/DM/train/train.amconll";
local train_data = "data/AMR/2015/train/train.amconll";
local dev_data = "data/AMR/2015/gold-dev/gold-dev.amconll";
#local dev_data = "data/SemEval/2015/DM/gold-dev/gold-dev.amconll";

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
    "model": {
        "type": "graph_dependency_parser",
        "edge_model" : {
            "type" : "kg_edges",
            "encoder_dim" : final_encoder_output_dim,
            "label_dim": 256,
            "edge_dim": 256,
            #"activation" : "tanh",
            "dropout": 0.1,
        },
        "dropout": 0.3,
        "input_dropout": 0.3,
        "encoder": {
            "type": "lstm",
            "bidirectional" : true,
            "num_layers" : 2,
            "hidden_size": encoder_output_dim,
            "input_size": 1024 + pos_dim + lemma_dim + ner_dim
        },
        "supertagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [1024],
                "dropout" : [0.3],
                "activations" : "tanh"
            }

        },
        "lexlabeltagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [1024],
                "dropout" : [0.3],
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
                        "dropout": 0.3
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
        "lexlabel_loss" : { }

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

