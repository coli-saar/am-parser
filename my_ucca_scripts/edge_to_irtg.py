import sys
def edge2irtg(edge_dict, id_label_dict):
    output = ''
    for key in list(edge_dict.keys()):
        if str(key[0]) +'/' + str(id_label_dict[key[0]]) in output:
            label_begin_edge = key[0]
        else:
            label_begin_edge = str(key[0]) +'/' +str(id_label_dict[key[0]])
        if str(key[1]) +'/' +str(id_label_dict[key[1]]) in output:
            label_end_edge = key[1]
        else:
            label_end_edge = str(key[1]) +'/' +str(id_label_dict[key[1]])
        edge = str(label_begin_edge) + ' -' + str(edge_dict[key]) + '-> ' + str(label_end_edge) + '; '
        output += edge
    new_format = '[' + output[0:-2] + ']'
    return new_format
