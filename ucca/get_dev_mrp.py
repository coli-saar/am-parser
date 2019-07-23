import sys
import json

mrp_everything = sys.argv[1]
dev_data = sys.argv[2]
out = sys.argv[3]


mrp_stored = {}
with open(mrp_everything) as mrp_in:
    for line in mrp_in:
        mrp = json.loads(line)
        id = mrp['id']
        mrp_stored[id] = mrp

#dev_stored = {}
dev_string = ''
with open(dev_data) as dev_in:
    for line in dev_in:
        if not line.startswith('#'):
            dev_string += line
dev_list = dev_string.split('\n\n')
dev_ids = [line.split('\n')[0] for line in dev_list]

with open(out, 'w') as outfile:
    for id in mrp_stored.keys():
        if id in dev_ids:
            outfile.write(json.dumps(mrp_stored[id]))
            outfile.write('\n')
