local ALTO_PATH = "/local/mlinde/alto-2.3-SNAPSHOT-jar-with-dependencies.jar";

local base_directory = "/local/mlinde/am-parser";

local tool_dir = base_directory + "/external_eval_tools/";

{
 "DM" : {
    "type" : "bash_evaluation_command",
    "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus --corpus {system_output} --gold {gold_file} --outFile /tmp/BLABLA && rm /tmp/BLABLA.sdp',
    "result_regexes" : { "F-Score" : [3, "F (?P<value>.+)"] } #says: on line 3 (0-based), fetch the F-Score with the given regex.
    },
  "PAS" : self.DM,
  "PSD" : {
    "type" : "bash_evaluation_command",
    "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPCorpus --corpus {system_output} --gold {gold_file} --outFile /tmp/BLABLA && rm /tmp/BLABLA.sdp',
    "result_regexes" : { "F-Score" : [3, "F (?P<value>.+)"] } #says: on line 3 (0-based), fetch the F-Score with the given regex.
    },

    "EDS" : { #don't use file extension for gold_file: use e.g. data/EDS/dev/dev-gold
    "type" : "bash_evaluation_command",
    "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus --corpus {system_output} --outFile /tmp/output.eds'+
    '&& python2 '+tool_dir+'/fast_smatch/fast_smatch.py -f output.eds.amr.txt {gold_file}.amr.txt --pr > /tmp/metrics.txt'+
    '&& python2 '+tool_dir+'/edm/eval_edm.py output.eds.edm {gold_file}.edm >> metrics.txt && cat /tmp/metrics.txt',
    "result_regexes" : {"Smatch-P" : [0, "Precision: (?P<value>.+)"],
                        "Smatch-R" : [1, "Recall: (?P<value>.+)"],
                        "Smatch-F" : [2, "F-score: (?P<value>.+)"],
                        "EDM-F" : [4,"F1-score: (?P<value>.+)"]}
    },

    "AMR-2015" : {
        "type" : "amr_evaluation_command",
        "amr_year" : "2015",
        "tool_dir" : tool_dir + "2019rerun",
        "alto_path" : ALTO_PATH,
    }
}

