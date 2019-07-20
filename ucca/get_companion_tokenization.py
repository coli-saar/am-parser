import sys
import json
import re
import os

companion_data_dir = sys.argv[1]
outdir = sys.argv[2]
hyphen_re = re.compile(r'[–—−]')
for filename in os.listdir(companion_data_dir):
    mrp_companion = {}
    with open(companion_data_dir+filename) as infile:
        for line in infile:
            if line.startswith('#'):
                id = line[1:].strip()
                mrp_companion[id] = {'tokenization':[], 'spans':{}}
            elif not line.startswith('\n'):
                line = line.split()
                token = line[1]
                token = re.sub(r'[–—−]', '-', token).lower()
                if "’" in token:
                    token = token.replace("’", "'")
                if "”" in token:
                    token = token.replace("”", '"')
                if "“" in token:
                    token = token.replace("“", '"')
                if "…" in token:
                    token = token.replace("…", "...")
                #if '-' in token:
                #    token = token.replace('-', '—')
                mrp_companion[id]['tokenization'].append(token)
                span = line[-1][11:]
                index = line[0]
                mrp_companion[id]['spans'][span] = int(index) -1
    json.dump(mrp_companion, open(outdir+ filename[:-7]+'.json', 'w', encoding='utf8'), ensure_ascii=False)
