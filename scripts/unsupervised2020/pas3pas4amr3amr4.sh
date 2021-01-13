# training PAS with the automata method, 3 sources
echo "starting PAS with 3 sources on GPU 0"
python -u train.py jsonnets/unsupervised2020/automata/PASautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/PASAuto3/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PAS auto3 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PASAuto3.log &

# training PAS with the automata method, 4 sources
echo "starting PAS with 4 sources on GPU 1"
python -u train.py jsonnets/unsupervised2020/automata/PASautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/PASAuto4/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PAS/train.zip"]], "validation_data_path": [["PAS", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/PAS/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PAS auto4 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PASAuto4.log &

# training AMR with the automata method, 3 sources
echo "starting AMR with 3 sources on GPU 2"
python -u train.py jsonnets/unsupervised2020/automata/AMRautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags AMR auto3 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3.log &

# training AMR with the automata method, 4 sources
echo "starting AMR with 4 sources on GPU 3"
python -u train.py jsonnets/unsupervised2020/automata/AMRautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto4/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/AMR/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags AMR auto4 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto4.log &

wait