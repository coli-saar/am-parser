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
import sys
def edge2irtg(edge_dict, id_label_dict):
    output = ''
    for key in list(edge_dict.keys()):
        if str(key[0]) +'/' + str(id_label_dict[key[0]]) in output.split():
            label_begin_edge = key[0]
        else:
            label_begin_edge = str(key[0]) +'/'+ str(id_label_dict[key[0]])
        if str(key[1]) +'/' +str(id_label_dict[key[1]]) in output.split():
            label_end_edge = key[1]
        else:
            label_end_edge = str(key[1]) +'/'+ str(id_label_dict[key[1]])
        edge = str(label_begin_edge) + ' -' + str(edge_dict[key]) + '-> ' + str(label_end_edge) + '; '
        output += edge
    new_format = '[' + output[0:-2] + ']'
    return new_format
