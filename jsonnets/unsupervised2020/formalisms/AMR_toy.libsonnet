
local ALTO_PATH = "am-tools.jar";
local WORDNET = "downloaded_models/wordnet3.0/dict/";
local tool_dir = "external_eval_tools/";

{
	"task": "AMR-2017",
	"evaluation_command" : {
		"type" : "bash_evaluation_command",
			"command" : "java -cp "+ALTO_PATH+" de.saar.coli.amtools.evaluation.EvaluateAMConll --corpus {system_output} -o {tmp}/ -ts AMREvaluationToolset -e '--wn "+WORDNET+
					" --lookup downloaded_models/lookup/lookupdata17/ --th 0 --add-sense-to-nn-label' " +
					"&& python2 "+tool_dir+"/smatch/smatch.py -f {tmp}/parserOut.txt {gold_file} --pr --significant 4 > {tmp}/metrics.txt && cat {tmp}/metrics.txt",
			"result_regexes" : {"P" : [0, "Precision: (?P<value>.+)"],
							"R" : [1, "Recall: (?P<value>.+)"],
							"F" : [2, "F-score: (?P<value>.+)"]}
	},
    "validation_metric" : "+AMR-2017_F",
}