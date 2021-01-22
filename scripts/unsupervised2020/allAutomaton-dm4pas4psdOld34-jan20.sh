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


# training PAS with the all automaton method, 4 sources
echo "starting PAS with 4 sources, all automaton, on GPU 0"
python -u train.py jsonnets/unsupervised2020/automata/PASallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PASAuto4-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }, "train_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PAS/train.zip"]], "validation_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PAS/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PAS allAutomaton4 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PASAuto4-allAutomaton-jan20.log &


# training DM with the all automaton method, 4 sources
echo "starting DM with 4 sources, all automaton, on GPU 1"
python -u train.py jsonnets/unsupervised2020/automata/DMallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/DMAuto4-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/DM/train.zip"]], "validation_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/DM/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags DM allAutomaton4 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/DMAuto4-allAutomaton-jan20.log &


# training PSD old pre with the all automaton method, 4 sources
echo "starting PSD with 4 sources, old pre, all automaton, on GPU 2"
python -u train.py jsonnets/unsupervised2020/automata/PSDallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDAuto4-oldPre-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDoldPre/train.zip"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDoldPre/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags oldPre allAutomaton4 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/PSDAuto4-oldPre-allAutomaton-jan20.log &


# training PSD old pre with the all automaton method, 3 sources
echo "starting PSD with 3 sources, old pre, all automaton, on GPU 3"
python -u train.py jsonnets/unsupervised2020/automata/PSDallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDAuto3-oldPre-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/PSDoldPre/train.zip"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/PSDoldPre/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags oldPre allAutomaton3 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/PSDAuto3-oldPre-allAutomaton-jan20.log &

wait