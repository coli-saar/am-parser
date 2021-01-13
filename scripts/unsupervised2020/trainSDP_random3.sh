amconllfolder="/proj/irtg/sempardata/unsupervised2020/amconll/random3"
modelprefix="random3"
comettag="random3"

modellocation="/local/jonasg/unsupervised2020/models"

mkdir -p $modellocation/$modelprefix/DM/
python -u train.py jsonnets/single/bert/DM.jsonnet -s $modellocation/$modelprefix/DM  -f --file-friendly-logging  -o "{\"trainer\" : {\"cuda_device\" :  0 }  , \"train_data_path\": [ [\"DM\", \"$amconllfolder/DM/train.amconll\"]] , \"validation_data_path\": [ [\"DM\", \"$amconllfolder/DM/dev.amconll\"]] }" --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags $comettag DM --project unsupervised2020 --workspace jgroschwitz &> $modellocation/$modelprefix/DM/log.txt &

mkdir -p $modellocation/$modelprefix/PAS/
python -u train.py jsonnets/single/bert/PAS.jsonnet -s $modellocation/$modelprefix/PAS  -f --file-friendly-logging  -o "{\"trainer\" : {\"cuda_device\" :  1 }  , \"train_data_path\": [ [\"PAS\", \"$amconllfolder/PAS/train.amconll\"]] , \"validation_data_path\": [ [\"PAS\", \"$amconllfolder/PAS/dev.amconll\"]] }" --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags $comettag PAS --project unsupervised2020 --workspace jgroschwitz &> $modellocation/$modelprefix/PAS/log.txt &

mkdir -p $modellocation/$modelprefix/PSD/
python -u train.py jsonnets/single/bert/PSD.jsonnet -s $modellocation/$modelprefix/PSD -f --file-friendly-logging  -o "{\"trainer\" : {\"cuda_device\" :  2 }  , \"train_data_path\": [ [\"PSD\", \"$amconllfolder/PSD/train.amconll\"]] , \"validation_data_path\": [ [\"PSD\", \"$amconllfolder/PSD/dev.amconll\"]] }" --comet Yt3xk2gaFeevDwlxSNzN2VUKh --tags $comettag PSD --project unsupervised2020 --workspace jgroschwitz &> $modellocation/$modelprefix/PSD/log.txt &

wait

