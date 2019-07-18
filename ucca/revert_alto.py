import sys
import re

before_revert = sys.argv[1]
after_revert = sys.argv[2]


with open(after_revert, 'w+') as outfile:
    with open(before_revert) as infile:
        alto = ''
        for line in infile:
            alto += line
        alto = alto.split('\n\n')
        print(alto[2])
        for graph in alto:
            graph = graph.split('\n')
            print(graph)
            for line in graph:
                if line == graph[-1]:
                    outfile.write(line)
                    outfile.write('\n')
                else:
                    line = line.replace("'","’")
                    line = line.replace("RIGHT_QUOTATION", "”")
                    line = line.replace("LEFT_QUOTATION","“")
                    line = line.replace("QUOTATION_MARK", '"')
                    line = line.replace("...", "…")
                    line = line.replace("LEFT_SQUARE_BRACKET", "[")
                    line = line.replace("RIGHT_SQUARE_BRACKET", "]")
                    line = line.replace("c_acute", "c")
                    outfile.write(line)
                    outfile.write('\n')
            outfile.write('\n')
