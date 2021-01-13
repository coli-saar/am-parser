#CHECKLIST:

# Tasks?
# Main task?
# Freda?
# Evaluate on test?
# Evaluate on the right test corpora?

local lr = 0.01;
local num_epochs = 200;
local patience = 1000;
local pos_dim = 2;
local lemma_dim = 2;
local word_dim = 8;
local ner_dim = 2;
local lexlabel_dim = 8;
local hidden_dim = 8;
local encoder_output_dim = hidden_dim; #encoder output dim, per direction, so total will be twice as large

local ALTO_PATH = "am-tools.jar";
local WORDNET = "downloaded_models/wordnet3.0/dict/";
local tool_dir = "external_eval_tools/";

#=========FREDA=========
local use_freda = 0; #0 = no, 1 = yes
#=======================

local final_encoder_output_dim = 2 * encoder_output_dim + use_freda * 2 * encoder_output_dim; #freda again doubles output dimension

#============TASKS==============
local my_task = "AMR-2017";
local corpus_path = "example/toyAMR/minimalshuffled.amconll";
local gold_dev_corpus_path = "example/toyAMR/minimalgold.txt";
#===============================

local dataset_reader =  {
        "type": "amconll",
         "token_indexers": {
            "tokens": {
              "type": "single_id",
              "lowercase_tokens": true
            }
        },
		"allow_copy_despite_sense": true
    };

local data_iterator = {
        "type": "same_formalism",
        "batch_size": 1,
        "formalisms" : [my_task]
    };

	
# copied from configs/task_models.libsonnet and adapted
local task_model(name,dataset_reader, data_iterator, final_encoder_output_dim, edge_model, edge_loss, label_loss) = {
    "name" : name,
    "dropout": 0.0,

    "output_null_lex_label" : true,

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
                "hidden_dims" : [hidden_dim],
                "dropout" : [0],
                "activations" : "tanh"
            },
            "label_namespace": name+"_supertag_labels"

        },
        "lexlabeltagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [hidden_dim],
                "dropout" : [0],
                "activations" : "tanh"
            },
            "label_namespace":name+"_lex_labels"

        },
		
		
		# alignment learning stuff
		"lexlabel_encoder": {
            "type" : "shared_split_encoder",
            "formalisms" : [name],
			"formalisms_without_tagging": [],
            "encoder": {
                "type": "stacked_bidirectional_lstm",
                "num_layers": 2, #TWO LAYERS, we don't use sesame street.
                "recurrent_dropout_probability": 0.0,
                "layer_dropout_probability": 0.0,
                "use_highway": false,
                "hidden_size": hidden_dim,
                "input_size": lexlabel_dim
            }
        },
		"lexlabel_embedding":  {
           "embedding_dim": lexlabel_dim,
           "vocab_namespace": name+"_lex_labels"
        },
		"learn_alignments": true,

        #LOSS:
        "loss_mixing" : {
            "edge_existence" : 0.0,
            "edge_label": 0.0,
            "supertagging": 0.0,
            "lexlabel": 1.0
        },
        "loss_function" : {
            "existence_loss" : { "type" : edge_loss, "normalize_wrt_seq_len": false}, #e.g. kg_edge_loss
            "label_loss" : {"type" : "dm_label_loss" , "normalize_wrt_seq_len": false} #TODO: remove dirty hack
        },

        "supertagger_loss" : { "normalize_wrt_seq_len": false },
        "lexlabel_loss" : { "normalize_wrt_seq_len": false },
		
		"lexlabelcopier" : {
			"type": "lemma_and_token",
			"mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [hidden_dim],
                "dropout" : [0],
                "activations" : "tanh"
            }
		},

        "validation_evaluator": {
			"type": "standard_evaluator",
			"formalism" : my_task,
			"system_input" : corpus_path,
			"gold_file": gold_dev_corpus_path,
			"use_from_epoch" : 0,
			"predictor" : {
                "type" : "amconll_predictor",
                "dataset_reader" : dataset_reader, #same dataset_reader as above.
                "data_iterator" : data_iterator, #same bucket iterator also for validation.
                "k" : 2,
                "threads" : 1,
                "give_up": 15,
                "evaluation_command" : {
					"type" : "bash_evaluation_command",
					"command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus --corpus {system_output} -o {tmp}/ --relabel --wn '+WORDNET+
						' --lookup downloaded_models/lookup/lookupdata17/ --th 0 --add-sense-to-nn-label' +
					'&& python2 '+tool_dir+'/smatch/smatch.py -f {tmp}/parserOut.txt {gold_file} --pr --significant 4 > {tmp}/metrics.txt && cat {tmp}/metrics.txt',
					"result_regexes" : {"P" : [0, "Precision: (?P<value>.+)"],
										"R" : [1, "Recall: (?P<value>.+)"],
										"F" : [2, "F-score: (?P<value>.+)"]}
				},
			}
		},
};
	
	
	
	
{
    "dataset_reader": dataset_reader,
    "iterator": data_iterator,
     "vocabulary" : {
            "min_count" : {
            "lemmas" : 7,
            "words" : 7
     }
     },
    "model": {
        "type": "graph_dependency_parser",

        "tasks" : [task_model(my_task, dataset_reader, data_iterator, final_encoder_output_dim, "kg_edges","kg_edge_loss","kg_label_loss")],

        "input_dropout": 0.0,
        "encoder": {
            "type" : if use_freda == 1 then "freda_split" else "shared_split_encoder",
            "formalisms" : [my_task],
			"formalisms_without_tagging": [],
            "task_dropout" : 0.0, #only relevant for freda
            "encoder": {
                "type": "stacked_bidirectional_lstm",
                "num_layers": 2, #TWO LAYERS, we don't use sesame street.
                "recurrent_dropout_probability": 0.0,
                "layer_dropout_probability": 0.0,
                "use_highway": false,
                "hidden_size": hidden_dim,
                "input_size": word_dim + pos_dim + lemma_dim + ner_dim
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
            "tokens": {
                    "type": "embedding",
                    "embedding_dim": word_dim
                },
        },

    },
    "train_data_path": [ [my_task, corpus_path]],
    "validation_data_path": [ [my_task, corpus_path]],


    #=========================EVALUATE ON TEST=================================
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
        "validation_metric" : "+AMR-2017_F",
		"num_serialized_models_to_keep" : 1
		}
}

