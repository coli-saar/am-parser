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
from typing import Dict


def flatten(d : Dict):
    """
    Flattens a dictionary and uses the path separated with _ to give unique key names.
    :param d:
    :return:
    """
    r = dict()
    agenda = [ (key,[],d) for key in d.keys()]
    while agenda:
        key,path,d = agenda.pop()
        if not isinstance(d[key],dict):
            r["_".join(path+[str(key)])] = d[key]
        else:
            for subkey in d[key].keys():
                agenda.append((subkey,path+[str(key)],d[key]))
    return r


def merge_dicts(x: Dict, prefix:str, y: Dict):
    r = dict()
    for k,v in x.items():
        r[k] = v
    for k,v in y.items():
        r[prefix+"_"+k] = v
    return r