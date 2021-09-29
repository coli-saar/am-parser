#!/usr/bin bash
##
## Copyright (c) 2021 Saarland University.
##
## This file is part of AM Parser
## (see https://github.com/coli-saar/am-parser/).
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##

# TODO: add comet API key

# NOTE: must have created dirs already (model):
# cd MODELDIR
# mkdir -p {train,train100}{Bert,Token}_{kg,kgrel}/{a,b,c,d,e}
MODELDIR="/local/piaw/cogs2021/models/earlyOctober"
# subdirs train100Bert_kg/a/

COMETAPIKEY="dummy comet key"  # todo add comet API key
COMETPROJECT="cogs2021earlyOctober"

# & at the end of the call to make training/prediction run in the background (4 runs on 4 GPUs simultaneously)
# try to utilize 4 GPUs fully with minimal waiting time for next iteration:
# first run the first 4 out of 5 reruns for each configuration
# then run the last, fifth, run for each configuration (there are 8 configurations: {train,train100}{Bert,Token}{kg,kgrel})
# step 1, train all models (8 * 5 = 40. On 4 PPUs, so 10 iterations, each taking <=12hrs probably)
# step 2, --> predictions in other script

printf "\n*** Start with training runs at %s\n" "$(date  +'%F %T %Z')"
echo "*** TRAINING rounds start (10 rounds with 4 parallel runs each)"

configs=("train100Bert_kgrel" "train100Token_kgrel" "trainBert_kgrel" "trainToken_kgrel" "train100Bert_kg" "train100Token_kg" "trainBert_kg" "trainToken_kg")
for current_config in "${configs[@]}"; do
  echo "TRAINING for $current_config : first 4 models (a,b,c,d)"
  python3 -u train.py jsonnets/cogs2021/COGS_"$current_config".jsonnet -s "$MODELDIR"/"$current_config"/a/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags "$current_config" run1 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  0  } }' &> "$MODELDIR"/"$current_config"/training_"$current_config"_a.log &
  python3 -u train.py jsonnets/cogs2021/COGS_"$current_config".jsonnet -s "$MODELDIR"/"$current_config"/b/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags "$current_config" run2 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  1  } }' &> "$MODELDIR"/"$current_config"/training_"$current_config"_b.log &
  python3 -u train.py jsonnets/cogs2021/COGS_"$current_config".jsonnet -s "$MODELDIR"/"$current_config"/c/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags "$current_config" run3 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  2  } }' &> "$MODELDIR"/"$current_config"/training_"$current_config"_c.log &
  python3 -u train.py jsonnets/cogs2021/COGS_"$current_config".jsonnet -s "$MODELDIR"/"$current_config"/d/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags "$current_config" run4 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  3  } }' &> "$MODELDIR"/"$current_config"/training_"$current_config"_d.log &
  wait
done

echo "TRAINING ( 9) 5th, i.e. e, model of train models"
# e   trainToken_kgrel   trainToken_kg  trainBert_kg   trainBert_kgrel
python3 -u train.py jsonnets/cogs2021/COGS_trainToken_kgrel.jsonnet -s "$MODELDIR"/trainToken_kgrel/e/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags trainToken_kgrel run5 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  0  } }' &> "$MODELDIR"/trainToken_kgrel/training_trainToken_kgrel_e.log &
python3 -u train.py jsonnets/cogs2021/COGS_trainToken_kg.jsonnet -s "$MODELDIR"/trainToken_kg/e/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags trainToken_kg run5 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  1  } }' &> "$MODELDIR"/trainToken_kg/training_trainToken_kg_e.log &
python3 -u train.py jsonnets/cogs2021/COGS_trainBert_kg.jsonnet -s "$MODELDIR"/trainBert_kg/e/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags trainBert_kg run5 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  2  } }' &> "$MODELDIR"/trainBert_kg/training_trainBert_kg_e.log &
python3 -u train.py jsonnets/cogs2021/COGS_trainBert_kgrel.jsonnet -s "$MODELDIR"/trainBert_kgrel/e/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags trainBert_kgrel run5 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  3  } }' &> "$MODELDIR"/trainBert_kgrel/training_trainBert_kgrel_e.log &
wait

echo "TRAINING (10) 5th, i.e. e, model of train100 models"
# e   train100Token_kgrel   train100Token_kg  train100Bert_kg   train100Bert_kgrel
python3 -u train.py jsonnets/cogs2021/COGS_train100Token_kgrel.jsonnet -s "$MODELDIR"/train100Token_kgrel/e/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags train100Token_kgrel run5 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  0  } }' &> "$MODELDIR"/train100Token_kgrel/training_train100Token_kgrel_e.log &
python3 -u train.py jsonnets/cogs2021/COGS_train100Token_kg.jsonnet -s "$MODELDIR"/train100Token_kg/e/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags train100Token_kg run5 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  1  } }' &> "$MODELDIR"/train100Token_kg/training_train100Token_kg_e.log &
python3 -u train.py jsonnets/cogs2021/COGS_train100Bert_kg.jsonnet -s "$MODELDIR"/train100Bert_kg/e/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags train100Bert_kg run5 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  2  } }' &> "$MODELDIR"/train100Bert_kg/training_train100Bert_kg_e.log &
python3 -u train.py jsonnets/cogs2021/COGS_train100Bert_kgrel.jsonnet -s "$MODELDIR"/train100Bert_kgrel/e/ -f --file-friendly-logging --comet "$COMETAPIKEY" --tags train100Bert_kgrel run5 --project "$COMETPROJECT" -o ' {"trainer" : {"cuda_device" :  3  } }' &> "$MODELDIR"/train100Bert_kgrel/training_train100Bert_kgrel_e.log &
wait


printf "\n*** Done with all training runs"
printf "\n*** Next step: getting predictions (allruns_predictions.sh)\n"

printf "\n*** Completed earlyOctober_allruns_training.sh   (End time: %s)\n" "$(date  +'%F %T %Z')"
