#CHECKLIST:

//Differences to ACL 2019 setup:
// - smaller batch_size
// - no embeddings for lemmas
// - more epochs

# Tasks?
# Main task?
# Freda?
# Evaluate on test?
# Evaluate on the right test corpora?

local num_epochs = 100;
local device = 0;
local pos_dim = 32;
local ner_dim = 16;

local test_evaluators = import '../../../configs/am_parser/test_evaluators.libsonnet';

local data_paths = import '../../../configs/data_paths.libsonnet';

local eval_commands = import '../../../configs/eval_commands.libsonnet';

local UD_banks = data_paths["UD_banks"];

local task_models = import '../../../configs/am_parser/uniformify_task_models.libsonnet';

local bert_model = "bert-large-uncased";

local encoder_output_dim = 256; #encoder output dim, per direction, so total will be twice as large

#=========FREDA=========
local use_freda = 0; #0 = no, 1 = yes
#=======================

local final_encoder_output_dim = 2 * encoder_output_dim + use_freda * 2 * encoder_output_dim; #freda again doubles output dimension

#============TASKS==============
local my_tasks = ["PAS"];
local main_task = "PAS"; #what validation metric to pay attention to.
#===============================

local dataset_reader = {
        "type": "amconll",
        "token_indexers": {
            "bert": {
              "type": "bert-pretrained",
              "pretrained_model": bert_model,
            }
        }
      };

local data_iterator (batch_size) = {
        "type": "same_formalism",
        "batch_size": batch_size,
        "formalisms" : my_tasks
    };

local train_iterator = data_iterator(16);
local dev_iterator = data_iterator(64);


{
    "dataset_reader": dataset_reader,
    "iterator": train_iterator,
    "validation_iterator" : dev_iterator,

    "model": {
        "type": "graph_dependency_parser",

        "tasks" : [task_models(task_name,dataset_reader, train_iterator,dev_iterator, final_encoder_output_dim, "kg_edges","kg_edge_loss","kg_label_loss") for task_name in my_tasks],

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
                "input_size": 1024 + pos_dim + ner_dim
            }
        },

        "pos_tag_embedding":  {
           "embedding_dim": pos_dim,
           "vocab_namespace": "pos"
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

    },
    "train_data_path": [ [task_name, data_paths["train_data"][task_name]] for task_name in my_tasks],
    "validation_data_path": [ [task_name,data_paths["gold_dev_data"][task_name]] for task_name in my_tasks],


    #=========================EVALUATE ON TEST=================================
    "evaluate_on_test" : true,
    "test_evaluators" : [test_evaluators(dataset_reader, dev_iterator)[main_task]], #when training is done, call evaluation on test sets with best model as described here.
    #==========================================================================

    "trainer": {
        "type" : "am-trainer",
        "num_epochs": num_epochs,
        "patience" : 10,
        "cuda_device": device,
        "optimizer": {
            "type": "adam",
        },
        "validation_metric" : eval_commands["validation_metric"][main_task],
        "num_serialized_models_to_keep" : 1
    }
}

