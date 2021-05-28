#CHECKLIST:

# Tasks?
# Main task?
# Freda?
# Evaluate on test?
# Evaluate on the right test corpora?

# todo: lots of 'change back' to dos for changed hyperparameters for debugging
local lr = 0.01; # 0.001
local num_epochs = 100; # 100
local patience = 20; # 10000
# # we don't have PoS-tags, lemmas or Named Entities in COGS
# local pos_dim = 32,
# local lemma_dim = 64,
# local ner_dim = 16,
local hidden_dim = 32; # 256
local hidden_dim_mlp = 64; # 1024

local batch_size = 1;  #32, todo change back
local k_supertags_evaldecoder = 4;  # 6, number of supertags to be used during decoding todo change back
local formalism_eval_from_epoch = 1;  # 6, (edit distance, exact match calculation) # todo change back to 6


#============EMBEDDINGS=========
local embedding_name = "tokens";  # "bert" or "tokens"  # main switch

local bert_model = "bert-large-uncased";  # "bert-base-uncased"

local bert_embedding_dim = 1024;  # bert-base-uncased: 768
local bert_token_indexer = {"bert": { "type": "bert-pretrained", "pretrained_model": bert_model}};
local bert_text_field_embedder = {
    "type": "basic",
    "allow_unmatched_keys" : true,
    "embedder_to_indexer_map": { "bert": ["bert", "bert-offsets"] },
    "token_embedders": {
        "bert" : {
            "type": "bert-pretrained",
            "pretrained_model" : bert_model,
        }
    },
};

local token_embedding_dim = 32;  # 128, 256, 512, 768(=3*256), train has 743 vocab, bert large 1024
local token_indexer = {"tokens": {"type": 'single_id'}};
local token_text_field_embedder = {
    "type": 'basic',
    "token_embedders": {
        "tokens": {
            "type": 'embedding',
            "embedding_dim": token_embedding_dim,
            "trainable": true
        }
    }
};

# (0) The word embedding switch, part 0: embedding_dimension
local getWordEmbeddingDim(name="tokens") =
    if name == "bert" then
        bert_embedding_dim
    else
        token_embedding_dim;
# (1) Word embedding switch, part 1: token_indexer
local getIndexer(name="tokens") =
    if name == "bert" then
        bert_token_indexer
    else
        token_indexer;

# (2) Word embedding switch, part 2: text_field_embedder
local getEmbedder(name="tokens") =
    if name == "bert" then
        bert_text_field_embedder
    else
        token_text_field_embedder;

#===============================

# path relative from this file
# local test_evaluators = import '../../configs/test_evaluators.libsonnet';
local eval_commands = import '../../configs/eval_commands.libsonnet';

local encoder_output_dim = hidden_dim; #encoder output dim, per direction, so total will be twice as large

#=========FREDA=========
local use_freda = 0; #0 = no, 1 = yes
#=======================

local final_encoder_output_dim = 2 * encoder_output_dim + use_freda * 2 * encoder_output_dim; #freda again doubles output dimension

#============TASKS==============
local my_task = "COGS";
local path_prefix = "/home/wurzel/HiwiAK/cogs2021/";
# local path_prefix = "/proj/irtg/sempardata/cogs/";
local train_zip_corpus_path = path_prefix + "amconll/train.zip"; # amconll/Auto3/train.zip
# local train_tsv_corpus_path = path_prefix + "COGS/data/train1k_noprim.tsv";  # not used
local dev_zip_corpus_path = path_prefix + "amconll/dev.zip";
local dev_amconll_corpus_path = path_prefix + "amconll/dp_dev.amconll";  # output of PrepareDevData.java
local dev_tsv_corpus_path = path_prefix + "small/dev10.tsv";
# local dev_tsv_corpus_path = path_prefix + "COGS/data/dev.tsv";
#===============================

local dataset_reader =  {
        "type": "amconll_automata",
         "token_indexers": getIndexer(embedding_name),  # embedding type relevant here
    };

local amconll_dataset_reader = {
        "type": "amconll_unannotated",
        "token_indexers": getIndexer(embedding_name),  # embedding type relevant here
      };

local data_iterator = {
        "type": "same_formalism",
        "batch_size": batch_size,
        "formalisms" : [my_task]
    };


# copied from configs/task_models.libsonnet and adapted
local task_model(name,dataset_reader, data_iterator, final_encoder_output_dim, edge_model, edge_loss, label_loss) = {
    "name" : name,
    "dropout": 0.0, #0.3 #todo change back

    "output_null_lex_label" : true,  # todo what is this?
    "all_automaton_loss": true,  # todo do this? or skip? right now doesn't learn lexlabel at all

    "edge_model" : {
            "type" : edge_model, #e.g. "kg_edges",
            "encoder_dim" : final_encoder_output_dim,
            "label_dim": hidden_dim,
            "edge_dim": hidden_dim,
            #"activation" : "tanh",
            #"dropout": 0.0,
            "edge_label_namespace" : name+"_head_tags"
        },
         "supertagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [hidden_dim_mlp],
                "dropout" : [0], # [0.4] todo change back
                "activations" : "tanh"
            },
            "label_namespace": name+"_supertag_labels"

        },
        "lexlabeltagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [hidden_dim_mlp],
                "dropout" : [0], # [0.4] todo change back
                "activations" : "tanh"
            },
            "label_namespace":name+"_lex_labels"

        },

        #LOSS:
        "loss_mixing" : {
            "edge_existence" : 1.0,
            "edge_label": 1.0,  # not present in DMautomata?
            "supertagging": 1.0,
            "lexlabel": 1.0
        },
        "loss_function" : {
            "existence_loss" : { "type" : edge_loss, "normalize_wrt_seq_len": false}, #e.g. kg_edge_loss
            "label_loss" : {"type" : "dm_label_loss" , "normalize_wrt_seq_len": false} #TODO: remove dirty hack
        },

        "supertagger_loss" : { "normalize_wrt_seq_len": false },
        "lexlabel_loss" : { "normalize_wrt_seq_len": false },

        "validation_evaluator": {
            "type": "standard_evaluator",
            "formalism" : my_task,
            "system_input" : dev_amconll_corpus_path, # only-token-amconll
            "gold_file": dev_tsv_corpus_path, # gold file in COGS format (tsv)
            "use_from_epoch" : formalism_eval_from_epoch,
            "predictor" : {
                "type" : "amconll_automata_predictor",
                "dataset_reader" : amconll_dataset_reader, #need to read amconll file here.
                "data_iterator" : data_iterator, #same bucket iterator also for validation.
                "k" : k_supertags_evaldecoder,  # number of supertags to be used during decoding
                "threads" : 1,
                "give_up": 15,  #15, time limit in seconds before retry parsing with k-1 supertags todo change back
                "evaluation_command" : eval_commands['commands'][my_task]
            }
        },
};



## === final full specification =====
{
    "dataset_reader": dataset_reader,
    #"validation_dataset_reader": amconll_dataset_reader,  # for prediction . breaks with utf-8 error (due to dev set being zip-folder?)
    "iterator": data_iterator,
     "vocabulary" : {
            "min_count" : {
            # "lemmas" : 7,
            "words" : 1  # 7, todo change back?
     }
     },
    "model": {
        "type": "graph_dependency_parser_automata",

        "tasks" : [task_model(my_task, dataset_reader, data_iterator, final_encoder_output_dim, "kg_edges","kg_edge_loss","kg_label_loss")],

        "input_dropout": 0.0, #0.3 todo change back
        "encoder": {
            "type" : if use_freda == 1 then "freda_split" else "shared_split_encoder",
            "formalisms" : [my_task],
            "formalisms_without_tagging": [],
            "task_dropout" : 0.0, #only relevant for freda
            "encoder": {
                "type": "stacked_bidirectional_lstm",
                "num_layers": 2, #TWO LAYERS, we don't use sesame street.
                "recurrent_dropout_probability": 0,#0.4, todo change back later
                "layer_dropout_probability": 0,#0.3 todo change back later
                "use_highway": false,
                "hidden_size": hidden_dim,
                "input_size": getWordEmbeddingDim(embedding_name), # + pos_dim + lemma_dim + ner_dim   # embedding type relevant here
            }
        },

//        "pos_tag_embedding":  {
//           "embedding_dim": pos_dim,
//           "vocab_namespace": "pos"
//        },
//        "lemma_embedding":  {
//           "embedding_dim": lemma_dim,
//           "vocab_namespace": "lemmas"
//        },
//         "ne_embedding":  {
//           "embedding_dim": ner_dim,
//           "vocab_namespace": "ner_labels"
//        },

        "text_field_embedder": getEmbedder(embedding_name),  # embedding type relevant here

    },
    "train_data_path": [ [my_task, train_zip_corpus_path]],
    "validation_data_path": [ [my_task, dev_zip_corpus_path]],


    #=========================EVALUATE ON TEST=================================
    #"evaluate_on_test" : false,  # todo do we want to evaluate on test?
    #"test_evaluators" : [test_evaluators(amconll_dataset_reader, data_iterator)[my_task]], #when training is done, call evaluation on test sets with best model as described here.
    # no eval on test:
    "evaluate_on_test" : false,
    "test_evaluators" : [],
    #==========================================================================

    "trainer": {
        "type" : "am-trainer",
        "num_epochs": num_epochs,
        "patience" : patience,
        "optimizer": {
            "type": "adam",
            "lr": lr,
        },
        "validation_metric" : "-COGS_EditDistance",  # todo change "+ExactMatch" ?
        # "validation_metric" : eval_commands['metric_names'][my_task],  # "+ExactMatch" ?
        "num_serialized_models_to_keep" : 1
    }
}

