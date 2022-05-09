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

# training AMR with the retrain method, 3 sources
echo "starting AMR retrain with 3 sources on GPU 0"
python -u train.py jsonnets/acl2019/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRretrain3-jan17/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }, "train_data_path": [["AMR-2017", "/local/jonasg/unsupervised2020/temp/AMRAuto3-jan15/AMR-2017_amconll_list_train_epoch55.amconll"]], "validation_data_path": [["AMR-2017", "/local/jonasg/unsupervised2020/temp/AMRAuto3-jan15/AMR-2017_amconll_list_dev_epoch55.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags retrain3 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRretrain3-jan17.log &

# training AMR with the EM method, 3 sources
echo "starting AMR EM with 3 sources on GPU 1"
python -u train.py jsonnets/acl2019/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/temp/AMR-EM3-jan17/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/EM3/AMR/train.amconll"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/EM3/AMR/corpus.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags EM3 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMR-EM3-jan17.log &

# training AMR with the EMzero method, 3 sources
echo "starting AMR EMzero with 3 sources on GPU 2"
python -u train.py jsonnets/acl2019/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/temp/AMR-EMzero3-jan17/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/EM_0iter3/AMR/train.amconll"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/EM_0iter3/AMR/corpus.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags EMzero3 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMR-EMzero3-jan17.log &

# training AMR with the random method, 3 sources
echo "starting AMR random with 3 sources on GPU 3"
python -u train.py jsonnets/acl2019/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/temp/AMR-random3-jan17/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/random3/AMR/train.amconll"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/random3/AMR/corpus.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags random3 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMR-random3-jan17.log &

wait