from nltk.corpus import treebank as tb
import sys
ids = tb.fileids()

for id in ids:
    wsj = tb.words(id)
    wsj = ' '.join(wsj)
    wsj = wsj.split(' . ')
    counter = 1
    for i, sent in enumerate(wsj):
        with open('UCCA_English-WSJ-master/WSJ_DIR/'+str(id[:-4])+'.'+str(counter)+'.mrg', 'w') as outfile:
            if i + 1!= len(wsj):
                to_write = sent +' .'
                outfile.write(to_write)
            else:
                outfile.write(sent)
        counter += 1
