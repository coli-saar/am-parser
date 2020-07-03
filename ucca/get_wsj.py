#
# Copyright (c) 2020 Saarland University.
#
# This file is part of AM Parser
# (see https://github.com/coli-saar/am-parser/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from nltk.corpus import treebank as tb
import sys
ids = tb.fileids()

for id in ids:
    wsj = tb.words(id)
    wsj = ' '.join(wsj)
    wsj = wsj.split(' . ')
    counter = 1
    for i, sent in enumerate(wsj):
        with open('UCCA_English-WSJ-master/WSJ_DIR/'+str(id[:-4])+'.'+str(counter)+'.mrg', 'w') as outfile:
            if i + 1!= len(wsj):
                to_write = sent +' .'
                outfile.write(to_write)
            else:
                outfile.write(sent)
        counter += 1
