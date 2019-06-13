local k = 6;
local give_up = 1800; #30 minutes
local eval_commands = import 'eval_commands.libsonnet';


local SDP_evaluator(dataset_reader, data_iterator, name, threads) = [
        [name+"_id", { #prefix used for evaluation metric
            "type": "standard_evaluator",
            "formalism" : name,
            "system_input" : "data/SemEval/2015/"+name+"/test.id/test.id.amconll",
            "gold_file": "data/SemEval/2015/"+name+"/test.id/en.id."+std.asciiLower(name)+".sdp",
            "predictor" : {
                    "type" : "amconll_predictor",
                    "dataset_reader" : dataset_reader,
                    "data_iterator" : data_iterator,
                    "k" : k,
                    "threads" : threads,
                    "give_up": give_up,
                    "evaluation_command" : eval_commands['commands'][name]
            }
        }],[name+"_ood",{ #prefix used for evaluation metric
                "type": "standard_evaluator",
                "formalism" : name,
                "system_input" : "data/SemEval/2015/"+name+"/test.ood/test.ood.amconll",
                "gold_file": "data/SemEval/2015/"+name+"/test.ood/en.ood."+std.asciiLower(name)+".sdp",
                "predictor" : {
                        "type" : "amconll_predictor",
                        "dataset_reader" : dataset_reader,
                        "data_iterator" : data_iterator,
                        "k" : k,
                        "threads" : threads,
                        "give_up": give_up,
                        "evaluation_command" : eval_commands['commands'][name]
                }
        }]

];

#Defines test set evaluators for the formalisms
#Since we have in-domain and out-of-domain test sets, each formalism gets a list of evaluators!
function (dataset_reader, data_iterator) {
  "AMR-2015" :  [ ["AMR-2015",{ #prefix used for evaluation metric
        "type": "standard_evaluator",
        "formalism" : "AMR-2015",
        "system_input" : "data/AMR/2015/test/test.amconll",
        "gold_file": "test",
        "predictor" : {
                "type" : "amconll_predictor",
                "dataset_reader" : dataset_reader,
                "data_iterator" : data_iterator,
                "k" : k,
                "threads" : 8,
                "give_up": give_up,
                "evaluation_command" : eval_commands['commands']['AMR-2015']
        }

  }]],
    "AMR-2017" :  [ ["AMR-2017",{ #prefix used for evaluation metric
        "type": "standard_evaluator",
        "formalism" : "AMR-2017",
        "system_input" : "data/AMR/2017/test/test.amconll",
        "gold_file": "test",
        "predictor" : {
                "type" : "amconll_predictor",
                "dataset_reader" : dataset_reader,
                "data_iterator" : data_iterator,
                "k" : k,
                "threads" : 8,
                "give_up": give_up,
                "evaluation_command" : eval_commands['commands']['AMR-2017']
        }

  }]],

    "DM" : SDP_evaluator(dataset_reader, data_iterator,"DM",6),

    "PAS" : SDP_evaluator(dataset_reader, data_iterator,"PAS",6),

    "PSD" : SDP_evaluator(dataset_reader, data_iterator,"PSD",6),

    "EDS" :  [
        ["EDS",{
        "type": "standard_evaluator",
        "formalism" : "EDS",
        "system_input" : "data/EDS/test/test.amconll",
        "gold_file": "data/EDS/test/test-gold",
        "use_from_epoch" : 1,
        "predictor" : {
                "type" : "amconll_predictor",
                "dataset_reader" : dataset_reader, #same dataset_reader as above.
                "data_iterator" : data_iterator, #same bucket iterator also for validation.
                "k" : k,
                "threads" : 6,
                "give_up": give_up, #try parsing only for 1 second, then retry with smaller k
                "evaluation_command" : eval_commands['commands']['EDS']
        }}]
     ]

}
