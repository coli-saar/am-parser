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
to_remove = ['*T*-1', '*T*-2']

for id in ids:
    wsj = tb.sents(id)
    counter = 1
    for sent in wsj:
        print(sent)
        for word in sent:
            if word.startswith('*') or word == '0' or word == '*':
                sent.remove(word)
        for mark in to_remove:
            while mark in sent:
                sent.remove(mark)
        print(sent)
        print('_'*20)
        with open('UCCA_English-WSJ-master/WSJ_DIR_2/'+str(id[:-4])+'.'+str(counter)+'.mrg', 'w') as outfile:
            outfile.write('\n'.join(sent))
        counter += 1
