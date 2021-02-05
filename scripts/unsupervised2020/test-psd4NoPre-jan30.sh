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

# training PSD no pre with the all automaton method, 4 sources
echo "starting PSD test set $suffix with 4 sources, no pre, all automaton, on GPU $gpu"
python -u train.py jsonnets/unsupervised2020/automata/PSDallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDAuto4-noPre-test$suffix$gpu-allAutomaton-jan30/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDnoPre/train.zip"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDnoPre/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags noPre test$suffix$gpu --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PSDAuto4-noPre-test$suffix$gpu-allAutomaton-jan30.log &

gpu=1

# training PSD no pre with the all automaton method, 4 sources
echo "starting PSD test set $suffix with 4 sources, no pre, all automaton, on GPU $gpu"
python -u train.py jsonnets/unsupervised2020/automata/PSDallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDAuto4-noPre-test$suffix$gpu-allAutomaton-jan30/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDnoPre/train.zip"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDnoPre/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags noPre test$suffix$gpu --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PSDAuto4-noPre-test$suffix$gpu-allAutomaton-jan30.log &


gpu=2

# training PSD no pre with the all automaton method, 4 sources
echo "starting PSD test set $suffix with 4 sources, no pre, all automaton, on GPU $gpu"
python -u train.py jsonnets/unsupervised2020/automata/PSDallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDAuto4-noPre-test$suffix$gpu-allAutomaton-jan30/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" : 2 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDnoPre/train.zip"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDnoPre/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags noPre test$suffix$gpu --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PSDAuto4-noPre-test$suffix$gpu-allAutomaton-jan30.log &


gpu=3

# training PSD no pre with the all automaton method, 4 sources
echo "starting PSD test set $suffix with 4 sources, no pre, all automaton, on GPU $gpu"
python -u train.py jsonnets/unsupervised2020/automata/PSDallAutomaton.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDAuto4-noPre-test$suffix$gpu-allAutomaton-jan30/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDnoPre/train.zip"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PSDnoPre/dev.zip"]], "evaluate_on_test": true}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags noPre test$suffix$gpu --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PSDAuto4-noPre-test$suffix$gpu-allAutomaton-jan30.log &


wait