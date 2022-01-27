
local ALTO_PATH = "am-tools.jar";

{
	"task": "PSD",
	"evaluation_command" : {
		"type" : "bash_evaluation_command",
		"command" : "java -cp "+ALTO_PATH+" de.saar.coli.amtools.evaluation.EvaluateAMConll --corpus {system_output} --gold {gold_file} --outPath {tmp}/PSD_out/ -et PSDEvaluationToolset",
		"result_regexes" : {
			 "P" : [1, "Precision (?P<value>.+)"],
			 "R" : [2, "Recall (?P<value>.+)"],
			 "F" : [3, "F (?P<value>.+)"] #says: on line 3 (0-based), fetch the F-Score with the given regex.
		},
	},
    "validation_metric" : "+PSD_F",
}