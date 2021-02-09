# checklist
# graphbank (search and replace)
# sources (search and replace)
# method in comment, echo
# GPU in echo, command (x1)
# command jsonnet

# command model directory (-s): type, method, sources, date, special
# command log file, like model directory

# command data paths train and dev
# command tags
# command comet project amr or sdp
# command special options

suffix=$1

# training PAS with the all automaton method, 3 sources
echo "starting PAS test set $suffix with 3 sources, all automaton, on GPU 0"
python -u train.py jsonnets/unsupervised2020/automata/PASallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PASAuto3-test$suffix-allAutomaton-jan23/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }, "train_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/PAS/train.zip"]], "validation_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/PAS/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PAS test$suffix --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PASAuto3-test$suffix-allAutomaton-jan23.log &


# training DM with the all automaton method, 3 sources
echo "starting DM test set $suffix with 3 sources, all automaton, on GPU 1"
python -u train.py jsonnets/unsupervised2020/automata/DMallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/DMAuto3-test$suffix-allAutomaton-jan23/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/DM/train.zip"]], "validation_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/DM/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags DM test$suffix --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/DMAuto3-test$suffix-allAutomaton-jan23.log &


# training PSD old pre with the all automaton method, 4 sources
echo "starting PSD test set $suffix with 4 sources, old pre, all automaton, on GPU 2"
python -u train.py jsonnets/unsupervised2020/automata/PSDallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDAuto4-oldPre-test$suffix-allAutomaton-jan23/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDoldPre/train.zip"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDoldPre/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags oldPre test$suffix --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PSDAuto4-oldPre-test$suffix-allAutomaton-jan23.log &


# training AMR with the all automaton method, 3 sources
echo "starting AMR test set $suffix with 3 sources, all automaton, on GPU 3"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-test$suffix-allAutomaton-jan23/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags test$suffix --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-test$suffix-allAutomaton-jan23.log &


wait