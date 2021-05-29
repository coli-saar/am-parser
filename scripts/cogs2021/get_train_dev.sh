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
# getting train and dev zip files for training the am-parser and also dev amconll file for evaluation
# todo make variables below command line parameters?

jar="am-tools.jar"

# cogsdatadir should contain the tsv files for train and dev
cogsdatadir="/home/wurzel/HiwiAK/cogs2021/small"
traintsvfile="train20_nonprim.tsv"
devtsvfile="dev10.tsv"

# amconlldir: where output (zip files and amconll file will be written to)
amconlldir="/home/wurzel/HiwiAK/cogs2021/amconll/"
devevalprefix="dp_dev"
sources=3

cd ~/HiwiAK/am-parser/

printf "\n-->Automata for train ($traintsvfile) and dev ($devtsvfile) in $cogsdatadir : will create zip files in $amconlldir: ...\n"
java -cp $jar de.saar.coli.amtools.decomposition.SourceAutomataCLICOGS --trainingCorpus $cogsdatadir/$traintsvfile --devCorpus $cogsdatadir/$devtsvfile --outPath $amconlldir --nrSources $sources --algorithm automata

printf "\n-->Prepare dev data for evaluation ($devevalprefix amconll file in $amconlldir will be created from $cogsdatadir/$devtsvfile) ...\n"
java -cp $jar de.saar.coli.amrtagging.formalisms.cogs.tools.PrepareDevData --corpus $cogsdatadir/$devtsvfile --outPath $amconlldir --prefix $devevalprefix

printf "\nDone!\n"
