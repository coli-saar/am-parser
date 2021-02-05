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
gpu=0

# training AMR with the all automaton method, 3 sources
echo "starting AMR test set all Edges $suffix with 3 sources, all automaton, on GPU $gpu"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-allEdges-test$gpu$suffix-allAutomaton-jan29/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRallEdges/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRallEdges/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags test$gpu$suffix --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-allEdges-test$gpu$suffix-allAutomaton-jan29.log &

gpu=1
# training AMR with the all automaton method, 3 sources
echo "starting AMR test set all Edges $suffix with 3 sources, all automaton, on GPU $gpu"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-allEdges-test$gpu$suffix-allAutomaton-jan29/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRallEdges/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRallEdges/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags test$gpu$suffix --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-allEdges-test$gpu$suffix-allAutomaton-jan29.log &

gpu=2
# training AMR with the all automaton method, 3 sources
echo "starting AMR test set all Edges $suffix with 3 sources, all automaton, on GPU $gpu"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-allEdges-test$gpu$suffix-allAutomaton-jan29/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRallEdges/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRallEdges/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags test$gpu$suffix --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-allEdges-test$gpu$suffix-allAutomaton-jan29.log &

gpu=3
# training AMR with the all automaton method, 3 sources
echo "starting AMR test set all Edges $suffix with 3 sources, all automaton, on GPU $gpu"
python -u train.py jsonnets/unsupervised2020/automata/AMRallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-allEdges-test$gpu$suffix-allAutomaton-jan29/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRallEdges/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRallEdges/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags test$gpu$suffix --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-allEdges-test$gpu$suffix-allAutomaton-jan29.log &


wait