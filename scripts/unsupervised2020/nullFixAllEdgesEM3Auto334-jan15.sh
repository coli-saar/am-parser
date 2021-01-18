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


# training AMR with the EM method, all edges, 3 sources
echo "starting AMR EM, all edges, with 3 sources on GPU 0"
python -u train.py jsonnets/acl2019/single/bert/AMR-2017.jsonnet -s /local/jonasg/unsupervised2020/temp/AMREM3-allEdges-jan18/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  0 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/EM3/AMRAllEdges/train.amconll"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/EM3/AMRAllEdges/dev.amconll"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags EM3 allEdges --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMREM3-allEdges-jan18.log 


# training AMR with the automata method, all edges, 3 sources
echo "starting AMR automata, all edges, with 3 sources on GPU 1"
python -u train.py jsonnets/unsupervised2020/automata/AMRautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-allEdges-jan18/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  1 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRAllEdges/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMRAllEdges/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags auto3 allEdges --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-allEdges-jan18.log


# training AMR with the automata method, 3 sources, null-fix
echo "starting AMR automata nullfix with 3 sources on GPU 2"
python -u train.py jsonnets/unsupervised2020/automata/AMRautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto3-nullfix-jan18/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  2 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto3/AMR/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags auto3 nullfix --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto3-nullfix-jan18.log &


# training AMR with the automata method, 4 sources, null-fix
echo "starting AMR automata nullfix with 4 sources on GPU 3"
python -u train.py jsonnets/unsupervised2020/automata/AMRautomata.jsonnet -s /local/jonasg/unsupervised2020/temp/AMRAuto4-nullfix-jan18/ -f --file-friendly-logging -o ' {"trainer" : {"cuda_device" :  3 }, "train_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/AMR/train.zip"]], "validation_data_path": [["AMR-2017", "/proj/irtg/sempardata/unsupervised2020/amconll/Auto4/AMR/dev.zip"]]}' --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags auto4 nullfix --project unsupervised2020-amr &> /proj/irtg/sempardata/unsupervised2020/logs/AMRAuto4-nullfix-jan18.log &

wait