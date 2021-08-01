# checklist
# graphbank (search and replace)
# sources (search and replace)
# method in comment, echo
# GPU in echo, command (x1)
# command jsonnet

# command model directory (-s): type, method, sources, special, date
# command log file, like model directory

# command data paths train and dev
# command tags
# command comet project amr or sdp
# command special options: directory (-s), log, -o, tags, echo


# training AMR with the automata method, 3 sources
echo "starting AMR with 3 sources on GPU 0"
python -u train.py jsonnets/unsupervised2020/automata/AMRautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-jan15/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0}, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/dev.zip"]], }' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags AMR auto3 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-jan15.log &

# training AMR with the automata method, 3 sources
echo "starting AMR with 3 sources on GPU 1, lr002"
python -u train.py jsonnets/unsupervised2020/automata/AMRautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-lr002-jan15/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1, "optimizer": {"lr": 0.002}}, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/dev.zip"]], }' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags AMR auto3 lr002 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-lr002-jan15.log &

# training AMR with the automata method, 3 sources
echo "starting AMR with 3 sources on GPU 2, lr0005"
python -u train.py jsonnets/unsupervised2020/automata/AMRautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-lr0005-jan15/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2, "optimizer": {"lr": 0.0005}}, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/dev.zip"]], }' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags AMR auto3 lr0005 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-lr0005-jan15.log &

# training AMR with the automata method, 3 sources
echo "starting AMR with 3 sources on GPU 3, lr0002"
python -u train.py jsonnets/unsupervised2020/automata/AMRautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-lr0002-jan15/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3, "optimizer": {"lr": 0.0002}}, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/dev.zip"]], }' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags AMR auto3 lr0002 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-lr0002-jan15.log &

wait