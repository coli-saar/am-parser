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