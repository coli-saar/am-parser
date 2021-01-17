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


# training PAS with the retrain method, 3 sources
echo "starting PAS retrain with 3 sources on GPU 0"
python -u train.py jsonnets/acl2019/single/bert/PAS.jsonnet -s /local/jonasg/unsupervised2020/temp/PASretrain3-jan17/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }, "train_data_path": [["PAS", "/local/jonasg/unsupervised2020/temp/PASAuto3-jan15/PAS_amconll_list_train_epoch58.amconll"]], "validation_data_path": [["PAS", "/local/jonasg/unsupervised2020/temp/PASAuto3-jan15/PAS_amconll_list_dev_epoch58.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PAS retrain3 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PASretrain3-jan17.log &


# training DM with the retrain method, 3 sources
echo "starting DM retrain with 3 sources on GPU 1"
python -u train.py jsonnets/acl2019/single/bert/DM.jsonnet -s /local/jonasg/unsupervised2020/temp/DMretrain3-jan17/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["DM", "/local/jonasg/unsupervised2020/temp/DMAuto3-jan15/DM_amconll_list_train_epoch56.amconll"]], "validation_data_path": [["DM", "/local/jonasg/unsupervised2020/temp/DMAuto3-jan15/DM_amconll_list_dev_epoch56.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags DM retrain3 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/DMretrain3-jan17.log &


# training PSD with the retrain method, 4 sources
echo "starting PSD retrain with 4 sources on GPU 2"
python -u train.py jsonnets/acl2019/single/bert/PSD.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDretrain4-jan17/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }, "train_data_path": [["PSD", "/local/jonasg/unsupervised2020/temp/PSDAuto4-jan15/PSD_amconll_list_train_epoch59.amconll"]], "validation_data_path": [["PSD", "/local/jonasg/unsupervised2020/temp/PSDAuto4-jan15/PSD_amconll_list_dev_epoch59.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PSD retrain4 --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PSDretrain4-jan17.log &


# training AMR with the retrain method, 4 sources
echo "starting AMR retrain with 4 sources on GPU 3"
python -u train.py jsonnets/acl2019/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRretrain4-jan17/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/local/jonasg/unsupervised2020/temp/AMRAuto4/AMR-2017_amconll_list_train_epoch57.amconll"]], "validation_data_path": [["AMR-2017", "/local/jonasg/unsupervised2020/temp/AMRAuto4/AMR-2017_amconll_list_dev_epoch57.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags retrain4 --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRretrain4-jan17.log &

wait