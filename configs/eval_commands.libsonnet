local ALTO_PATH = "/local/mlinde/am-tools/build/libs/am-tools-all.jar";

local MTOOL = "/local/mlinde/mtool/main.py";
local base_directory = "/local/mlinde/am-parser";

local tool_dir = base_directory + "/external_eval_tools/";

local sdp_regexes = {
 "P" : [1, "Precision (?P<value>.+)"],
 "R" : [2, "Recall (?P<value>.+)"],
 "F" : [3, "F (?P<value>.+)"] #says: on line 3 (0-based), fetch the F-Score with the given regex.

};

{
 "commands" : { #commands to evaluate the different formalisms
     "DM" : {
        "type" : "bash_evaluation_command",
        "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus --corpus {system_output} --gold {gold_file} --outFile {tmp}/BLABLA',
        "result_regexes" : sdp_regexes
        },
      "PAS" : self.DM,
      "PSD" : {
        "type" : "bash_evaluation_command",
        "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPCorpus --corpus {system_output} --gold {gold_file} --outFile {tmp}/BLABLA',
        "result_regexes" : sdp_regexes
        },

        "EDS" : { #don't use file extension for gold_file: use e.g. data/EDS/dev/dev-gold
            "type" : "bash_evaluation_command",
            "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus --corpus {system_output} --outFile {tmp}/output.eds'+
            '&& python2 '+tool_dir+'/fast_smatch/fast_smatch.py -f {tmp}/output.eds.amr.txt {gold_file}.amr.txt --pr > {tmp}/metrics.txt'+
            '&& python2 '+tool_dir+'/edm/eval_edm.py {tmp}/output.eds.edm {gold_file}.edm >> {tmp}/metrics.txt && cat {tmp}/metrics.txt',
            "result_regexes" : {"Smatch_P" : [0, "Precision: (?P<value>.+)"],
                                "Smatch_R" : [1, "Recall: (?P<value>.+)"],
                                "Smatch_F" : [2, "F-score: (?P<value>.+)"],
                                "EDM_F" : [4,"F1-score: (?P<value>.+)"]}
        },

        "AMR-2015" : {
            "type" : "amr_evaluation_command",
            "amr_year" : "2015",
            "tool_dir" : tool_dir + "2019rerun",
            "alto_path" : ALTO_PATH,
        },

         "AMR-2017" : {
            "type" : "amr_evaluation_command",
            "amr_year" : "2017",
            "tool_dir" : tool_dir + "2019rerun",
            "alto_path" : ALTO_PATH,
        },

        "MRP-DM" : {
        "type" : "json_evaluation_command",
        "commands" : [["",'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateMRP --corpus {system_output} --out {tmp}/output.mrp'],
                        ["sdp",'python3 '+MTOOL+' --read mrp --score sdp --gold {gold_file} {tmp}/output.mrp'],
                        ["mrp",'python3 '+MTOOL+' --read mrp --score mrp --limit 10000 --gold {gold_file} {tmp}/output.mrp']]
        },

        "MRP-PSD" : {
        "type" : "json_evaluation_command",
        "commands" : [["",'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateMRP --corpus {system_output} --out {tmp}/output.mrp'],
                        ["sdp",'python3 '+MTOOL+' --read mrp --score sdp --gold {gold_file} {tmp}/output.mrp'],
                        ["mrp",'python3 '+MTOOL+' --read mrp --score mrp --limit 10000 --gold {gold_file} {tmp}/output.mrp']]
        },


    },

    "metric_names": { #the name and direction of each validation metric for each formalism, + means "higher is better"
        "DM" : "+DM_F",
        "PAS": "+PAS_F",
        "PSD": "+PSD_F",
        "EDS": "+EDS_Smatch_F",
        "AMR-2015": "+AMR-2015_F-score",
        "AMR-2017": "+AMR-2017_F-score",

        "MRP-DM" : "+MRP-DM_mrp_all_f",
        "MRP-PSD" : "+MRP-PSD_mrp_all_f"

    }



}

