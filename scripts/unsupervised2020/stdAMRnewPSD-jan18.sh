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


# training original PSD (new pre/post) 
echo "starting PSD (orig heuristics; new pre/post) on GPU 2"
python -u train.py jsonnets/acl2019/single/bert/PSD.jsonnet -s /local/jonasg/unsupervised2020/temp/PSDheuristicsNewPrePost-jan18/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }, "train_data_path": [["PSD", "/proj/irtg/sempardata/uniformify2020/new_psd_preprocessing/train/train.amconll"]], "validation_data_path": [["PSD", "/proj/irtg/sempardata/uniformify2020/new_psd_preprocessing/gold-dev/gold-dev.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags PSD newprepost --project unsupervised2020 &> /proj/irtg/sempardata/unsupervised2020/logs/PSDheuristicsNewPrePost-jan18.log &


# training original AMR
echo "starting AMR (orig heuristics) on GPU 3"
python -u train.py jsonnets/acl2019/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRheuristics-jan18/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/ACL2019/AMR/2017/train/train.amconll"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/ACL2019/AMR/2017/gold-dev/gold-dev.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags ACL19redo --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRheuristics-jan18.log &

wait