local num_epochs = 40;
local device = 1;
local pos_dim = 32;
local lemma_dim = 64;
local ner_dim = 16;

local validation_evaluators = import 'validation_evaluators.libsonnet';

local test_evaluators = import 'test_evaluators.libsonnet';

local data_paths = import 'data_paths.libsonnet';

local eval_commands = import 'eval_commands.libsonnet';

local UD_banks = ["EWT","GUM","LinES","ParTUT"];

local elmo_path = "/local/mlinde/elmo/";
local encoder_output_dim = 128; #encoder output dim, per direction, so total will be twice as large
local use_freda = 0; #0 = no, 1 = yes
local final_encoder_output_dim = 2 * encoder_output_dim + use_freda * 2 * encoder_output_dim; #freda again doubles output dimension


local my_tasks = ["AMR-2017"];# + UD_banks;
local main_task = "AMR-2017"; #what validation metric to pay attention to.

local dataset_reader = {
        "type": "amconll",
        "token_indexers": {
            "elmo": {
              "type": "elmo_characters"
            }
        }
      };
local data_iterator = {
        "type": "same_formalism",
        "batch_size": 64,
        "formalisms" : my_tasks
    };


local task(name) = { #defines the part of the model that is specific for each task.
    name : name,
    "dropout": 0.3,

    "edge_model" : {
            "type" : "kg_edges",
            "encoder_dim" : final_encoder_output_dim,
            "label_dim": 256,
            "edge_dim": 256,
            #"activation" : "tanh",
            "dropout": 0.0,
            "edge_label_namespace" : name+"_head_tags"
        },
         "supertagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [1024],
                "dropout" : [0.4],
                "activations" : "tanh"
            },
            "label_namespace": name+"_supertag_labels"

        },
        "lexlabeltagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [1024],
                "dropout" : [0.4],
                "activations" : "tanh"
            },
            "label_namespace":name+"_lex_labels"

        },

        #LOSS:
        "loss_mixing" : {
            "edge_existence" : 1.0,
            "edge_label": 1.0,
            "supertagging": if std.count(UD_banks,name) > 0 then null else 1.0, #disable supertagging for UD
            "lexlabel": if std.count(UD_banks,name) > 0 then null else 1.0
        },
        "loss_function" : {
            "existence_loss" : { "type" : "kg_edge_loss", "normalize_wrt_seq_len": false},
            "label_loss" : {"type" : "dm_label_loss" , "normalize_wrt_seq_len": false}
        },

        "supertagger_loss" : { "normalize_wrt_seq_len": false },
        "lexlabel_loss" : { "normalize_wrt_seq_len": false },

        "validation_evaluator": validation_evaluators(dataset_reader, data_iterator)[name]
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

        "tasks" : [task(task_name) for task_name in my_tasks],

        "input_dropout": 0.3,
        "encoder": {
            "type" : if use_freda == 1 then "freda_split" else "shared_split_encoder",
            "formalisms" : my_tasks,
            "formalisms_without_tagging": UD_banks,
            "task_dropout" : 0.0,
            "encoder": {
                "type": "stacked_bidirectional_lstm",
                "num_layers": 1,
                "recurrent_dropout_probability": 0.4,
                "layer_dropout_probability": 0.3,
                "use_highway": false,
                "hidden_size": encoder_output_dim,
                "input_size": 1024 + pos_dim + lemma_dim + ner_dim
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
                        "dropout": 0.1
                    },
             }
        },

    },
    "train_data_path": [ [task_name, data_paths["train_data"][task_name]] for task_name in my_tasks],
    "validation_data_path": [ [task_name,data_paths["gold_dev_data"][task_name]] for task_name in my_tasks],

    "evaluate_on_test" : true,
    "test_evaluators" : [test_evaluators(dataset_reader, data_iterator)[main_task]], #when training is done, call evaluation on test sets with best model as described here.

    "trainer": {
        "type" : "am-trainer",
        "num_epochs": num_epochs,
        "patience" : 10,
        "cuda_device": device,
        "optimizer": {
            "type": "adam",
        },
        "validation_metric" : eval_commands["metric_names"][main_task],
        "num_serialized_models_to_keep" : 1
    }
}

