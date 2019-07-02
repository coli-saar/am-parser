import os
import sys
import xml.etree.ElementTree as et

source_directory = 'xml/'
token_directory = 'WSJ_DIR_2/'

for filename in os.listdir(source_directory):
    id = filename[:-4]
    xml_file = source_directory + filename
    tree = et.parse(xml_file)
    #find text with same id
    for token_filename in os.listdir(token_directory):
        id_token = token_filename[:-4]
        if id == id_token:
            text = ' '
            with open(token_directory+token_filename) as infile:
                for line in infile:
                    text += line
                    #print(text)
            nodes_and_text = list(zip(tree.getroot()[1][1:], text.split()))
            print(nodes_and_text)
            for (node, text) in nodes_and_text:
                for attribute in node:
                    print(node, attribute, text)
                    attribute.attrib['text'] = text
    print('________________________________________')
    tree.write('xml_output/'+filename)
