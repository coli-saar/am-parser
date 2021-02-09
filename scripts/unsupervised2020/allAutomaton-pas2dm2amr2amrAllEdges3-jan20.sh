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


# training PAS with the all automaton method, 2 sources
echo "starting PAS with 2 sources, all automaton, on GPU 0"
python -u train.py jsonnets/unsupervised2020/automata/PASallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PASAuto2-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }, "train_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto2/PAS/train.zip"]], "validation_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto2/PAS/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PAS allAutomaton2 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PASAuto2-allAutomaton-jan20.log &


# training DM with the all automaton method, 2 sources
echo "starting DM with 2 sources, all automaton, on GPU 1"
python -u train.py jsonnets/unsupervised2020/automata/DMallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/DMAuto2-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto2/DM/train.zip"]], "validation_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto2/DM/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags DM allAutomaton2 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/DMAuto2-allAutomaton-jan20.log &


# training AMR with the all automaton method, 2 sources
echo "starting AMR with 2 sources, all automaton, on GPU 2"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto2-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto2/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto2/AMR/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags allAutomaton2 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto2-allAutomaton-jan20.log &


# training AMR with the all automaton method, 3 sources
echo "starting AMR with 3 sources, all edges, all automaton, on GPU 3"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-allEdges-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRAllEdges/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRAllEdges/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags allEdges allAutomaton3 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-allEdges-allAutomaton-jan20.log &

wait