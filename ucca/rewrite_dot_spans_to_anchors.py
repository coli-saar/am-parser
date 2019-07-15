import re
import json
import sys

dot_file = sys.argv[1]
mrp_file = sys.argv[2]
identifier = str(sys.argv[3])

span_regex = re.compile(r"〈\d+:\d+〉")

span_to_anchor = {}
with open(mrp_file) as infile:
    for line in infile:
        mrp = json.loads(line)
        print(mrp['id'], ' ', identifier)

        if mrp['id'] == identifier:
            input_string = mrp['input']
            for node in mrp['nodes']:
                if 'anchors' in node.keys():
                    for anchor in node['anchors']:
                        span_to_anchor['〈'+str(anchor['from'])+':'+str(anchor['to'])+'〉'] = input_string[anchor['from']:anchor['to']]
print(span_to_anchor)
with open(dot_file) as infile:
        dot = ''
        for line in infile:
            dot += line
        for span in span_to_anchor.keys():
            dot = dot.replace(span, span_to_anchor[span])
        print(dot)
        with open(identifier+'_anchored'+'.dot', 'w') as outfile:
            outfile.write(dot)
