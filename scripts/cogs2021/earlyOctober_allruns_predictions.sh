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

# run this after earlyOctober_allruns_train.sh has finished successfully
# (trained models need to exist)
# NOTE: must have created dirs already (pred, cogs, evalsummary):
# cd PREDICTIONDIR
# mkdir -p {gen,test}/{train,train100}{Bert,Token}_{kg,kgrel}/{a,b,c,d,e}
MODELDIR="/local/piaw/cogs2021/models/earlyOctober"
# subdirs train100Bert_kg/a/
PREDICTIONDIR="/local/piaw/cogs2021/predictions/earlyOctober"
# subdirs gen / train100Bert_kg/a/
COGSDATADIR="/proj/irtg/sempardata/cogs2021/data/COGS/data"  # todo pull most recent version and remember commit
EVALSUMMARYDIR="/proj/irtg/sempardata/cogs2021/earlyOctober/summary_results"

# & at the end of the call to make training/prediction run in the background (4 runs on 4 GPUs simultaneously)
# try to utilize 4 GPUs fully with minimal waiting time for next iteration:
# first run the first 4 out of 5 reruns for each configuration
# then run the last, fifth, run for each configuration (there are 8 configurations: {train,train100}{Bert,Token}{kg,kgrel})
# step 1, train all models --> done in the other script
# step 2, predict using trained models on gen and on test: this should be rather fast: less than an hour for an iteration?
# step 3, summarize results in files: last 55 lines of prediction with --verbose should be enough (unless many ill-formed)

configs=("train100Bert_kgrel" "train100Token_kgrel" "trainBert_kgrel" "trainToken_kgrel" "train100Bert_kg" "train100Token_kg" "trainBert_kg" "trainToken_kg")

printf "\n*** Start predicting with all models at %s\n" "$(date  +'%F %T %Z')"
# all models with Astar (`-p` option)
corpora=("gen" "test")
for corpus in "${corpora[@]}"; do
  echo "*** Starting predictions for all models on $corpus :"

  # prediction simultaneously with the first four models of every configuration
  for modelconfig in "${configs[@]}"; do
    echo "PREDICTION for $corpus with $modelconfig : first 4 models (a,b,c,d)"
    bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/"$modelconfig"/a -m "$MODELDIR"/"$modelconfig"/a/model.tar.gz -g 0 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_"$modelconfig"_a.log &
    bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/"$modelconfig"/b -m "$MODELDIR"/"$modelconfig"/b/model.tar.gz -g 1 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_"$modelconfig"_b.log &
    bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/"$modelconfig"/c -m "$MODELDIR"/"$modelconfig"/c/model.tar.gz -g 2 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_"$modelconfig"_c.log &
    bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/"$modelconfig"/d -m "$MODELDIR"/"$modelconfig"/d/model.tar.gz -g 3 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_"$modelconfig"_d.log &
    wait
  done

  # remaining 8 (5th model for each of the configurations)
  # configs=("train100Bert_kgrel" "train100Token_kgrel" "trainBert_kgrel" "trainToken_kgrel" "train100Bert_kg" "train100Token_kg" "trainBert_kg" "trainToken_kg")
  echo "PREDICTION for $corpus ( 9): remaining kgrel models (models e)"
  bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/train100Bert_kgrel/e -m "$MODELDIR"/train100Bert_kgrel/e/model.tar.gz -g 0 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_train100Bert_kgrel_e.log &
  bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/train100Token_kgrel/e -m "$MODELDIR"/train100Token_kgrel/e/model.tar.gz -g 1 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_train100Token_kgrel_e.log &
  bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/trainBert_kgrel/e -m "$MODELDIR"/trainBert_kgrel/e/model.tar.gz -g 2 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_trainBert_kgrel_e.log &
  bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/trainToken_kgrel/e -m "$MODELDIR"/trainToken_kgrel/e/model.tar.gz -g 3 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_trainToken_kgrel_e.log &
  wait
  echo "PREDICTION for $corpus (10): remaining kg models (models e)"
  bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/train100Bert_kg/e -m "$MODELDIR"/train100Bert_kg/e/model.tar.gz -g 0 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_train100Bert_kg_e.log &
  bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/train100Token_kg/e -m "$MODELDIR"/train100Token_kg/e/model.tar.gz -g 1 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_train100Token_kg_e.log &
  bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/trainBert_kg/e -m "$MODELDIR"/trainBert_kg/e/model.tar.gz -g 2 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_trainBert_kg_e.log &
  bash ./scripts/cogs2021/unsupervised_predict.sh -i "$COGSDATADIR"/"$corpus".tsv -o "$PREDICTIONDIR"/"$corpus"/trainToken_kg/e -m "$MODELDIR"/trainToken_kg/e/model.tar.gz -g 3 -p &> "$PREDICTIONDIR"/"$corpus"/"$corpus"_trainToken_kg_e.log &
  wait

  summaryFile="$EVALSUMMARYDIR"/"$corpus"_55summary.txt
  true > "$summaryFile"  # get empty file
  for modelconfig in "${configs[@]}"; do
    for runchar in a b c d e; do
      printf "\n\nEVALUATION RESULTS $corpus %s $runchar :\n" "$modelconfig" >> "$summaryFile"
      tail -n 55 "$PREDICTIONDIR"/"$corpus"/"$corpus"_"$modelconfig"_"$runchar".log >> "$summaryFile"
    done
  done

  echo "*** Done with all predictions for $corpus"
done

printf "\n*** Completed earlyOctober_allruns_predictions.sh   (End time: %s)\n" "$(date  +'%F %T %Z')"
