local ALTO_PATH = "am-tools.jar";

local WORDNET = "downloaded_models/wordnet3.0/dict/";

local CONCEPTNET = "/proj/irtg.shadow/data/conceptnet-assertions-5.7.0.csv.gz";


local MTOOL = "/proj/irtg.shadow/tools/mtool/main.py";
local base_directory = "/local/mlinde/am-parser";

local tool_dir = base_directory + "/external_eval_tools/";

local data_paths = import 'data_paths.libsonnet';
local MRP_AMR_SUBPATH = data_paths["MRP_AMR_SUBPATH"];

local sdp_regexes = {
 "P" : [1, "Precision (?P<value>.+)"],
 "R" : [2, "Recall (?P<value>.+)"],
 "F" : [3, "F (?P<value>.+)"] #says: on line 3 (0-based), fetch the F-Score with the given regex.

};

{

/*
 This allows us to check if the dependencies seem to be met at start of training.
*/
  "am-tools" : ALTO_PATH,

  "extra_dependencies" : {
    "AMR-2015" : [WORDNET],
    "AMR-2017" : [WORDNET],
    "MRP-DM" : [MTOOL],
    "MRP-PSD" : [MTOOL],
    "MRP-EDS" : [MTOOL],
    "MRP-AMR" : [CONCEPTNET, MTOOL],
    "MRP-UCCA" : [MTOOL]
  },

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
            "type" : "bash_evaluation_command",
            "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus --corpus {system_output} -o {tmp}/ --relabel --wn '+WORDNET+
                ' --lookup data/AMR/2015/lookup/ --th 10' +
            '&& python2 '+tool_dir+'/smatch/smatch.py -f {tmp}/parserOut.txt {gold_file} --pr --significant 4 > {tmp}/metrics.txt && cat {tmp}/metrics.txt',
            "result_regexes" : {"P" : [0, "Precision: (?P<value>.+)"],
                                "R" : [1, "Recall: (?P<value>.+)"],
                                "F" : [2, "F-score: (?P<value>.+)"]}
        },

        "AMR-2017" : {
            "type" : "bash_evaluation_command",
            "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus --corpus {system_output} -o {tmp}/ --relabel --wn '+WORDNET+
                ' --lookup data/AMR/2017/lookup/ --th 10' +
            '&& python2 '+tool_dir+'/smatch/smatch.py -f {tmp}/parserOut.txt {gold_file} --pr --significant 4 > {tmp}/metrics.txt && cat {tmp}/metrics.txt',
            "result_regexes" : {"P" : [0, "Precision: (?P<value>.+)"],
                                "R" : [1, "Recall: (?P<value>.+)"],
                                "F" : [2, "F-score: (?P<value>.+)"]}
        },

        "AMR-2020" : {
            "type" : "bash_evaluation_command",
            "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus --corpus {system_output} -o {tmp}/ --relabel --wn '+WORDNET+
                ' --lookup data/AMR/2020/lookup/ --th 10' +
            '&& python2 '+tool_dir+'/smatch/smatch.py -f {tmp}/parserOut.txt {gold_file} --pr --significant 4 > {tmp}/metrics.txt && cat {tmp}/metrics.txt',
            "result_regexes" : {"P" : [0, "Precision: (?P<value>.+)"],
                                "R" : [1, "Recall: (?P<value>.+)"],
                                "F" : [2, "F-score: (?P<value>.+)"]}
        },

        "MRP-DM" : {
        "type" : "json_evaluation_command",
        "commands" : [["",'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateMRP --corpus {system_output} --out {tmp}/output.mrp'],
                        ["sdp",'python3 '+MTOOL+' --read mrp --score sdp --gold {gold_file} {tmp}/output.mrp'],
                        ["mrp",'python3 '+MTOOL+' --read mrp --score mrp --cores 4 --limit 10000 --gold {gold_file} {tmp}/output.mrp']]
        },

        "MRP-PSD" : {
        "type" : "json_evaluation_command",
        "commands" : [["",'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateMRP --corpus {system_output} --out {tmp}/output.mrp'],
                        ["sdp",'python3 '+MTOOL+' --read mrp --score sdp --gold {gold_file} {tmp}/output.mrp'],
                        ["mrp",'python3 '+MTOOL+' --read mrp --score mrp --cores 4 --limit 10000 --gold {gold_file} {tmp}/output.mrp']]
        },

        "MRP-EDS" : {
        "type" : "json_evaluation_command",
        "commands" : [["",'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateMRP --corpus {system_output} --out {tmp}/output.mrp'],
                        ["edm",'python3 '+MTOOL+' --read mrp --score edm --gold {gold_file} {tmp}/output.mrp'],
                        ["smatch",'python3 '+MTOOL+' --read mrp --score smatch --limit 2 --gold {gold_file} {tmp}/output.mrp'],
                        ]
                        #["mrp",'python3 '+MTOOL+' --read mrp --score mrp --limit 10000 --gold {gold_file} {tmp}/output.mrp']]
        },

        "MRP-AMR" : {
        "type" : "json_evaluation_command",
        "commands" : [["",'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateAMR --conceptnet '+CONCEPTNET +' --wn external_eval_tools/2019rerun/metadata/wordnet/3.0/dict/ --lookup data/MRP/AMR/'+MRP_AMR_SUBPATH+'/lookup/ --corpus {system_output} --out {tmp}/output.mrp'],
                        ["smatch",'python3 '+MTOOL+' --read mrp --score smatch --cores 4 --gold {gold_file} {tmp}/output.mrp'],
                        ["mrp",'python3 '+MTOOL+' --read mrp --score mrp --limit 3 --cores 4 --gold {gold_file} {tmp}/output.mrp'],
                        ]
        },

        "MRP-UCCA" : {
        "type" : "json_evaluation_command",
        "commands" : [["",'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateMRP --corpus {system_output} --out {tmp}/output.mrp'],
                        ["","python3 ucca/decompress_mrp.py {tmp}/output.mrp {tmp}/output_post.mrp"],
                        ["","python3 ucca/remove_labels.py {tmp}/output_post.mrp {tmp}/output_post_no_labels.mrp"],
                        ["ucca",'python3 '+MTOOL+' --read mrp --score ucca --gold {gold_file} {tmp}/output_post_no_labels.mrp'],
                        ["mrp",'python3 '+MTOOL+' --read mrp --score mrp --cores 4 --limit 2 --gold {gold_file} {tmp}/output_post_no_labels.mrp'],
                        ]
        },

    },

    "metric_names": { #the name and direction of each validation metric for each formalism, + means "higher is better"
        "DM" : "+DM_F",
        "PAS": "+PAS_F",
        "PSD": "+PSD_F",
        "EDS": "+EDS_Smatch_F",
        "AMR-2015": "+AMR-2015_F",
        "AMR-2017": "+AMR-2017_F",
        "AMR-2020": "+AMR-2020_F",

        "MRP-DM" : "+MRP-DM_mrp_all_f",
        "MRP-PSD" : "+MRP-PSD_mrp_all_f",
        "MRP-EDS" : "+MRP-EDS_smatch_f",
        "MRP-AMR" : "+MRP-AMR_mrp_all_f",

        "MRP-UCCA" : "+MRP-UCCA_mrp_all_f",

        "EWT" : "+EWT_LAS",
        "GUM" : "+GUM_LAS",
        "LinES" : "+LinES_LAS",
        "ParTUT" : "+ParTUT_LAS"

    },

    #MRP postprocessing command instead of full evaluation command because we don't have gold graphs:
    "postprocessing" : {
        "MRP-DM" : ['java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateMRP --corpus {system_output} --out {system_output}.mrp --input data/MRP/test/input.mrp'],
        "MRP-PSD" : ['java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateMRP --corpus {system_output} --out {system_output}.mrp --input data/MRP/test/input.mrp'],
        "MRP-EDS" : ['java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateMRP --corpus {system_output} --out {system_output}.mrp --input data/MRP/test/input.mrp'],
        "MRP-UCCA" : ['java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateMRP --corpus {system_output} --out {system_output}.mrp --input data/MRP/test/input.mrp',
                       "python3 ucca/decompress_mrp.py {system_output}.mrp {system_output}.post.mrp",
                       "python3 ucca/remove_labels.py {system_output}.post.mrp {system_output}.post.nolabels.mrp"],
        "MRP-AMR" : ['java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.mrp.tools.EvaluateAMR --conceptnet '+CONCEPTNET +' --wn external_eval_tools/2019rerun/metadata/wordnet/3.0/dict/ --lookup data/MRP/AMR/'+MRP_AMR_SUBPATH+'/lookup/ --corpus {system_output} --out {system_output}.mrp --input data/MRP/test/input.mrp']

    }



}

