# model parameters
local lr = 0.01; # learning rate
local pos_dim = 2; # embedding dimension for POS (part of speech) tags
local lemma_dim = 2; # embedding dimension for lemmas
local word_dim = 8; # embedding dimension for the words themselves
local ner_dim = 2; # embedding dimension for named entity tags
local hidden_dim = 8; # hidden layer dimension
local final_encoder_output_dim = 2 * hidden_dim; # gets one input per encoder direction, so total will be twice as large
local k = 4;  # number of supertags per word to be used during evaluation (i.e. using top-k supertags).
local eval_threads = 1;
local eval_timeout = 15;
	
local dataset_reader =  {
	"type": "amconll_automata",
	 "token_indexers": {
		"tokens": {
		  "type": "single_id",
		  "lowercase_tokens": true
		}
	},
};
	
local amconll_dataset_reader =  {
	"type": "amconll_unannotated",
	 "token_indexers": {
		"tokens": {
		  "type": "single_id",
		  "lowercase_tokens": true
		}
	}
};

local iterator(batch_size, task) = {
	"type": "same_formalism",
	"batch_size": batch_size,
	"formalisms" : [task]
};


local task_model(task,dataset_reader, iterator, final_encoder_output_dim, edge_model, edge_loss, label_loss, evaluation_command, validation_amconll_path, validation_gold_path, batch_size) = {
    "name" : task,
    "dropout": 0.0,

    "output_null_lex_label" : true,
	
	"all_automaton_loss": true,

    "edge_model" : {
            "type" : edge_model, #e.g. "kg_edges",
            "encoder_dim" : final_encoder_output_dim,
            "label_dim": hidden_dim,
            "edge_dim": hidden_dim,
            #"activation" : "tanh",
            #"dropout": 0.0,
            "edge_label_namespace" : task+"_head_tags"
        },
         "supertagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [hidden_dim],
                "dropout" : [0],
                "activations" : "tanh"
            },
            "label_namespace": task+"_supertag_labels"

        },
        "lexlabeltagger" : {
            "mlp" : {
                "input_dim" : final_encoder_output_dim,
                "num_layers" : 1,
                "hidden_dims" : [hidden_dim],
                "dropout" : [0],
                "activations" : "tanh"
            },
            "label_namespace":task+"_lex_labels"

        },

        #LOSS:
        "loss_mixing" : {
            "edge_existence" : 1.0,
            "edge_label": 1.0,
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
			"formalism" : task,
			"system_input" : validation_amconll_path,
			"gold_file": validation_gold_path,
			"use_from_epoch" : 10,
			"predictor" : {
                "type" : "amconll_automata_predictor",
                "dataset_reader" : amconll_dataset_reader, #need to read the amconll file here, not the zip.
                "data_iterator" : iterator(batch_size, task), #same bucket iterator also for validation.
                "k" : k,
                "threads" : eval_threads,
                "give_up": eval_timeout,
                "evaluation_command" : evaluation_command,
			}
		},
};


local make_test_evaluator(task, evaluation_command, batch_size, test_triple_amconll_gold_suffix) = {
	"result" : [task+test_triple_amconll_gold_suffix[2],{ #prefix used for evaluation metric
        "type": "standard_evaluator",
        "formalism" : task,
        "system_input" : test_triple_amconll_gold_suffix[0],
        "gold_file": test_triple_amconll_gold_suffix[1],
        "predictor" : {
                "type" : "amconll_predictor",
                "dataset_reader" : amconll_dataset_reader,
                "data_iterator" : iterator(batch_size, task),
                "k" : k,
                "threads" : eval_threads,
                "give_up": eval_timeout,
                "evaluation_command" : evaluation_command,
        }
  }]
};


function(batch_size, num_epochs, patience, task, evaluation_command, validation_metric, validation_amconll_path, validation_gold_path, test_triples_amconll_gold_suffix) {
	
	"dataset_reader": dataset_reader,
	"amconll_dataset_reader": amconll_dataset_reader,
    "iterator": iterator(batch_size, task),
	"vocabulary" : {
            "min_count" : {
            "lemmas" : 7,
            "words" : 7
		}
    },
	
	"model": {
        "type": "graph_dependency_parser_automata",

        "tasks" : [task_model(task, dataset_reader, iterator, final_encoder_output_dim, "kg_edges","kg_edge_loss","kg_label_loss", evaluation_command, validation_amconll_path, validation_gold_path, batch_size)],

        "input_dropout": 0.0,
        "encoder": {
            "type" : "shared_split_encoder",
            "formalisms" : [task],
			"formalisms_without_tagging": [],
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
	
	
    "trainer": {
        "type" : "am-trainer",
        "num_epochs": num_epochs,
        "patience" : patience,
        "optimizer": {
            "type": "adam",
			"lr": lr,
        },
        "validation_metric" : validation_metric,
		"num_serialized_models_to_keep" : 1
		},
	
	"test_evaluators": [std.map(function(x) make_test_evaluator(task, evaluation_command, batch_size, x)['result'], test_triples_amconll_gold_suffix)],
}
