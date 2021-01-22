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


# training PAS with the all automaton method, 3 sources
echo "starting PAS with 3 sources, all automaton, on GPU 0"
python -u train.py jsonnets/unsupervised2020/automata/PASallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PASAuto3-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }, "train_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/PAS/train.zip"]], "validation_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/PAS/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PAS allAutomaton3 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PASAuto3-allAutomaton-jan20.log &


# training DM with the all automaton method, 3 sources
echo "starting DM with 3 sources, all automaton, on GPU 1"
python -u train.py jsonnets/unsupervised2020/automata/DMallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/DMAuto3-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/DM/train.zip"]], "validation_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/DM/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags DM allAutomaton3 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/DMAuto3-allAutomaton-jan20.log &


# training AMR with the all automaton method, 3 sources
echo "starting AMR with 3 sources, all automaton, on GPU 2"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags allAutomaton3 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-allAutomaton-jan20.log &


# training AMR with the all automaton method, 4 sources
echo "starting AMR with 4 sources, all automaton, on GPU 3"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto4-allAutomaton-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/AMR/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags allAutomaton4 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto4-allAutomaton-jan20.log &

wait