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


# training PSD old pre with the all automaton method, 2 sources
echo "starting PSD with 2 sources, old pre, all automaton, on GPU 0"
python -u train.py jsonnets/unsupervised2020/automata/PSDallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDAuto2-oldPre-allAutomaton-jan23/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto2/PSDoldPre/train.zip"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto2/PSDoldPre/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags oldPre allAutomaton2 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PSDAuto2-oldPre-allAutomaton-jan23.log &


# training PSD old pre with the all automaton method, 3 sources
echo "starting PSD with 3 sources, old pre, all automaton, on GPU 1"
python -u train.py jsonnets/unsupervised2020/automata/PSDallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDAuto3-oldPre-allAutomaton-jan23/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/PSDoldPre/train.zip"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/PSDoldPre/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags oldPre allAutomaton3 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PSDAuto3-oldPre-allAutomaton-jan23.log &

# training AMR with the all automaton method, 2 sources
echo "starting AMR with 2 sources, all edges, all automaton, on GPU 2"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto2-allEdges-allAutomaton-jan23/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto2/AMRallEdges/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto2/AMRallEdges/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags allEdges allAutomaton2 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto2-allEdges-allAutomaton-jan23.log &

# training AMR with the all automaton method, 4 sources
echo "starting AMR with 4 sources, all edges, all automaton, on GPU 3"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto4-allEdges-allAutomaton-jan23/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/AMRallEdges/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/AMRallEdges/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags allEdges allAutomaton4 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto4-allEdges-allAutomaton-jan23.log &


wait