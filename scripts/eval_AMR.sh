#!/bin/bash
##
## Copyright (c) 2020 Saarland University.
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
ALTO_PATH=$(readlink -f "$2")

# if these files are missing, run scripts/setup_AMR.sh
lookup="downloaded_models/lookup/lookupdata17/"
wordnet_path="downloaded_models/wordnet3.0/dict/"

echo "relabelling with threshold 10"

# run the relabeller
java -Xmx2G -cp "$ALTO_PATH" de.saar.coli.amrtagging.formalisms.amr.tools.Relabel $dir/ "$lookup" "$wordnet_path" 10

echo "final post-processing"

# now we have a relabelled.txt in this directory
# do more post-processing
sed -E 's/\(u_[0-9]+ \/ ([-+0-9]+)\)/\1/g' $dir/relabeled.txt | sed -E 's/\(explicitanon[0-9]+ \/ ([^"()]+)\)/"\1"/g' | sed -E 's/\(explicitanon[0-9]+ \/ ("[^"]+")\)/\1/g' | sed -E 's/"([-+0-9]+)"/\1/g' > $dir/relabeled_fixed.txt

# compute smatch score
echo "smatching in $dir"

python2 evaluation_tools/smatch/smatch.py -f $dir/relabeled_fixed.txt $dir/gold_orderedAsRelabeled.txt --significant 4 --pr | tee $dir/smatch.txt
