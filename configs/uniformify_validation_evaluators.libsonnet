local k = 6;
local eval_commands = import 'eval_commands.libsonnet';
local give_up = 15; #15 seconds

local data_paths = import 'data_paths.libsonnet';
local MRP_AMR_SUBPATH = data_paths["MRP_AMR_SUBPATH"];
local MRP_UCCA_SUBPATH = data_paths["MRP_UCCA_SUBPATH"];
local SDP_prefix = data_paths["SDP_prefix"];

local sdp_evaluator(dataset_reader, data_iterator, name, threads, from_epoch) = {
        "type": "standard_evaluator",
        "formalism" : name,
        "system_input" : SDP_prefix+name+"/dev/dev.amconll",
        "gold_file": SDP_prefix+name+"/dev/dev.sdp",
        "use_from_epoch" : from_epoch,
        "predictor" : {
                "type" : "amconll_predictor",
                "dataset_reader" : dataset_reader, #same dataset_reader as above.
                "data_iterator" : data_iterator, #same bucket iterator also for validation.
                "k" : k,
                "threads" : threads,
                "give_up": give_up,
                "evaluation_command" : eval_commands['commands'][name]
        }

};


#Defines validation evaluators for the formalisms
function (dataset_reader, data_iterator) {


    "DM" : sdp_evaluator(dataset_reader, data_iterator, "DM", 8, 25),

    "PAS" :  sdp_evaluator(dataset_reader, data_iterator, "PAS", 8, 25),

    "PSD" :  sdp_evaluator(dataset_reader, data_iterator, "PSD", 8, 25),


}
