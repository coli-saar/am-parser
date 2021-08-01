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

# training AMR with the retrain method, 4 sources
echo "starting AMR retrain with 4 sources on GPU 3"
python -u train.py jsonnets/acl2019/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRretrain4-jan17/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/local/jonasg/unsupervised2020/temp/AMRAuto4/AMR-2017_amconll_list_train_epoch57.amconll"]], "validation_data_path": [["AMR-2017", "/local/jonasg/unsupervised2020/temp/AMRAuto4/AMR-2017_amconll_list_dev_epoch57.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags retrain4 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRretrain4-jan17.log 
