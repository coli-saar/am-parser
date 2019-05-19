import os
import sys

if "ALTO_PATH" not in os.environ:
    print("Please set the environment variable ALTO_PATH to point to Alto .jar (generalized am, with dependencies)")
    sys.exit(-1)

ALTO_PATH = "/local/mlinde/alto-2.3-SNAPSHOT-jar-with-dependencies.jar"

SDP = 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus --corpus {1} --gold {0}.sdp --outFile {1}.sdp > {1}.txt'

THIS_FILE_LIVES_IN = "/".join(os.path.realpath(__file__).split("/")[:-1])

eval_commands = {
 "DM" : [SDP], "PAS" : [SDP], "PSD" : [SDP], "SDP" : [SDP],
    "EDS" : ['java -cp '+os.environ["ALTO_PATH"]+' de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus --corpus {1} --outFile {1}',
             #'python2 ' +THIS_FILE_LIVES_IN+"/eval/smatch/smatch.py --pr --significant 3 -f {1}.amr.txt {0}-gold.amr.txt > {1}.txt", #official smatch - slow
             'python2 ' +THIS_FILE_LIVES_IN+"/eval/fast_smatch/fast_smatch.py --pr -f {1}.amr.txt {0}-gold.amr.txt  > {1}.txt", #fast_smatch by Liu et al. - reimplementation with cython and C++
             "python2 "+THIS_FILE_LIVES_IN+"/eval/edm/eval_edm.py {1}.edm {0}-gold.edm >> {1}.txt"
             ]
}

def read_sdp(filename):
    ret = []
    with open(filename) as f:
        next(f) #skip first line
        for _ in range(4):
            line = next(f).split(" ")
            ret.append((line[0], 100 * float(line[-1])))
    return ret

def read_eds(filename):
    ret = []
    with open(filename) as f:
        for _ in range(3):
            line = next(f).split(" ")
            ret.append(("Smatch "+line[0].rstrip(":"), 100 * float(line[1])))
        next(f)
        line = next(f).split(" ")
        ret.append(("EDM "+line[0].rstrip(":"),float(line[1])))
    return ret

read_scores = {
    "DM" : read_sdp, "PAS" :  read_sdp, "PSD" : read_sdp, "SDP" : read_sdp, "EDS": read_eds
}

def evaluate(taskname, system_output, gold_data):
    """
    Expects a task name, the filename out the system output and the filename (without extension!) of the gold data. Here are two examples:

    print(evaluate("SDP","models/deleteme/dev_epoch_SDP-DM-2015_1.amconll","data/SemEval/2015/DM/dev/dev"))
    print(evaluate("EDS","models/deleteme/dev_epoch_EDS_1.amconll","data/EDS/dev/dev"))

    Again, note that the file for the gold data has no extension.

    :param taskname:
    :param system_output:
    :param gold_data:
    :return: a dictionary that maps metrics to their respective values
    """
    for command in eval_commands[taskname]:
        print("Waiting for the following command to terminate:",command.format(gold_data,system_output))
        os.system(command.format(gold_data,system_output))
    ret = []
    for command in eval_commands[taskname]:
        ret.extend(read_scores[taskname] (system_output+".txt"))
    return dict(ret)


