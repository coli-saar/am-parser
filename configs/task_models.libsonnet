local validation_evaluators = import 'validation_evaluators.libsonnet';

#defines the part of the model that is specific for each task.

local data_paths = import 'data_paths.libsonnet';
local UD_banks = data_paths["UD_banks"];

#those graph banks where it is desired to have lexical labels that are _
#or more general: including a graphbank in this list will always make the parser return the most likely lexical label
local OUTPUT_NULL_LEX_LABEL = ["DM","PSD","PAS","MRP-UCCA"];

function(name,dataset_reader, data_iterator, final_encoder_output_dim, edge_model, edge_loss, label_loss) {
    "name" : name,
    "dropout": 0.3,

    "output_null_lex_label" : std.count(OUTPUT_NULL_LEX_LABEL,name) > 0,

    "edge_model" : {
            "type" : edge_model, #e.g. "kg_edges",
            "encoder_dim" : final_encoder_output_dim,
            "label_dim": 256,
            "edge_dim": 256,
            #"activation" : "tanh",
            #"dropout": 0.0,
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
            "existence_loss" : { "type" : edge_loss, "normalize_wrt_seq_len": false}, #e.g. kg_edge_loss
            "label_loss" : {"type" : "dm_label_loss" , "normalize_wrt_seq_len": false} #TODO: remove dirty hack
        },

        "supertagger_loss" : { "normalize_wrt_seq_len": false },
        "lexlabel_loss" : { "normalize_wrt_seq_len": false },

        "validation_evaluator": validation_evaluators(dataset_reader, data_iterator)[name]
}