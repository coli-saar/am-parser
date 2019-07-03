from nltk.corpus import treebank as tb
import sys
ids = tb.fileids()
to_remove = ['*T*-1', '*T*-2']

for id in ids:
    wsj = tb.sents(id)
    counter = 1
    for sent in wsj:
        print(sent)
        for word in sent:
            if word.startswith('*') or word == '0' or word == '*':
                sent.remove(word)
        for mark in to_remove:
            while mark in sent:
                sent.remove(mark)
        print(sent)
        print('_'*20)
        with open('UCCA_English-WSJ-master/WSJ_DIR_2/'+str(id[:-4])+'.'+str(counter)+'.mrg', 'w') as outfile:
            outfile.write('\n'.join(sent))
        counter += 1
