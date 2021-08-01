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


# training PAS with the automata method, 6 sources
echo "starting PAS with 6 sources on GPU 4"
python -u train.py jsonnets/unsupervised2020/automata/PASautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/PASAuto6-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  4 }, "train_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto6/PAS/train.zip"]], "validation_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto6/PAS/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PAS auto6 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PASAuto6-jan20.log &


# training DM with the automata method, 6 sources
echo "starting DM with 6 sources on GPU 5"
python -u train.py jsonnets/unsupervised2020/automata/DMautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/DMAuto6-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  5 }, "train_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto6/DM/train.zip"]], "validation_data_path": [["DM", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto6/DM/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags DM auto6 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/DMAuto6-jan20.log &


# training PSD with the automata method, 6 sources
echo "starting PSD with 6 sources on GPU 6"
python -u train.py jsonnets/unsupervised2020/automata/PSDautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDAuto6-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  6 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto6/PSD/train.zip"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto6/PSD/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PSD auto6 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PSDAuto6-jan20.log &


# training AMR with the automata method, 6 sources
echo "starting AMR with 6 sources on GPU 7"
python -u train.py jsonnets/unsupervised2020/automata/AMRautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto6-jan20/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  7 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto6/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto6/AMR/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags auto6 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto6-jan20.log &

wait