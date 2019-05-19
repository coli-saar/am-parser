# cython: language_level=3

from operator import itemgetter
import graph_dependency_parser.am_algebra.tree as tree
import tqdm
from graph_dependency_parser.am_algebra.amconll import ConllEntry, ConllSent, write_conll
from graph_dependency_parser.components.cle import get_head_dict

from time import time

import pyximport; pyximport.install()
import graph_dependency_parser.am_algebra.amtypes as amtypes
#import graph_dependency_parser.am_algebra.evaluator as evaluator
import multiprocessing as mp
import numpy as np

class AgendaItem:
    """
    Represents a parse item that lives on the agenda.
    """
    def __init__(self,index,unproc_children,type,score,local_types,subtree):
        self.index = index
        self.unproc_children = unproc_children
        self.type = type
        self.score = score
        self.local_types = local_types
        self.subtree = subtree
    def __str__(self):
        return "AgendaItem({},{},{},{},{},{})".format(self.index,self.unproc_children,self.type,self.score,self.local_types,self.subtree)
    def __repr__(self):
        return str(self)
    def __le__(self,other):
        return True
    def __ge__(self,other):
        return True
    def __lt__(self,other):
        return True


class AMDecoder:
    
    def __init__(self,output_file, i2edgelabel):
        """

        :param output_file: The filename where to store the decoded trees (*.amconll)
        :param i2edgelabel: A list of edge names that - by their order - allows to map indices to edge labels
        """
        self.totyp = dict() #look up table to avoid parsing the same type from string to python object twice
        self.output_file = output_file
        self.i2rel = i2edgelabel
        self.rel2i = { rel : i for i, rel in enumerate(i2edgelabel)}

        self.sents = []
    

    def add_sentence(self, root : int, heads, label_scores, lexlabels, supertagpreds, sentence, attributes):
        """
        Add a new sentence to our todo list of sentences to parse.
        In the following, let N be the length of the sentence, L the number of different labels
        :param root: index of the root of dependency tree
        :param heads: a numpy array of length N, the head of every word.
        :param label_scores: a numpy array for edge label scores with dimensions (N+1) x L. label_scores[i,l] represents the score for the incoming edge to i having label l
        :param lexlabels: a list of of lexical labels (strings) that will be used for the words
        :param supertagpreds: a list of lists. For each word, this list contains triples (log probability, graph fragment as string, AM type as string) and must contain an entry for \bot (represented by _)
        :param sentence: a list of tuples in the shape: (form,replacement,lemma,pos,ne)
        :param attributes: a list of sentence-wide attributes, e.g. ["raw:This is the untokenized sentence"]
        :return: None
        """

        conllsent = ConllSent([-1]+list(heads), label_scores, root)
        for attr in attributes:
            conllsent.add_attr(attr)
        root_entry = ConllEntry(0, '*root*', '*root*', '*root*', '*root*', '*root*', 'ROOT-FRAGMENT', '*root*', 'ROOT-TYPE',
                          -1, 'rroot')
        conllsent.append(root_entry)
        assert len(sentence) == len(lexlabels)
        assert len(sentence) == len(supertagpreds)
        for i,(form,replacement,lemma,pos,ne) in enumerate(sentence):
            entry = ConllEntry(i+1,form,replacement,lemma,pos,ne,"_",lexlabels[i],"_",0,"IGNORE",True)
            entry.pred_parent_id = heads[i]
            if not any(triple[2] == "_" for triple in supertagpreds[i]):
                raise ValueError("The supertag prediction for word "+str(i+1)+" has no entry for bottom (represented by _). This is required.")
            entry.supertags = sorted(supertagpreds[i], key= lambda triple: triple[0], reverse=True)
            conllsent.append(entry)
        assert len(conllsent) == len(heads)+1

        self.sents.append((conllsent,))


    def decode(self, threads, kbest, giveup):
        """
        Actually parse the sentences. Write the results to the output file.
        :param threads:
        :param kbest:
        :param giveup: time limit for each sentence until we back off to lower k
        :return:
        """
        self.kbest = kbest
        self.giveupat = giveup
        if self.giveupat:
            print("Will give up after",self.giveupat,"s when still not successful")
        print("Using",kbest,"best supertags")
        if threads > 1:
            with mp.Pool(threads) as pool:
                r = list(pool.starmap(self.call_viterbi,self.sents))
        else:
            tqdm_sents = tqdm.tqdm(self.sents)
            r = [ self.call_viterbi(s[0], tqdm_obj=tqdm_sents) for s in tqdm_sents]
        write_conll(self.output_file,r)
        self.sents = []










        
    def parse_am_type(self, string):
        
        if string in self.totyp:
            return self.totyp[string]
        self.totyp[string] = amtypes.parse_type(string)
        return self.totyp[string]
    
    def get_items(self, entry, kbest):
        """
            Creates a list of Items for a (leaf) entry. Assumes entry to have a list called supertags where the tags are sorted in descending order by score.
        """
        ms = []
        types_used = []
        null_item = None
        for (s,delex,typ) in entry.supertags:
            t = self.parse_am_type(typ)
            if not t in types_used:
                ms.append(AgendaItem(entry.id,set(),t,s,[(entry.id,t)],[]))
                types_used.append(t)
            if t == self.parse_am_type("_"):
                null_item = AgendaItem(entry.id,set(),t,s,[(entry.id,t)],[])
        best_ones = ms[0:kbest]
        if null_item is None:
            raise ValueError("It looks like there was no prediction for the type \\bot (written _ here) provided")
        if not null_item in best_ones:
            best_ones.append(null_item)
        return best_ones
    

    
    def call_viterbi(self, conll_sentence, tqdm_obj=None):
        assert(all(len(entry.supertags) >= self.kbest for entry in conll_sentence[1:]))

        return self.viterbi(conll_sentence,conll_sentence.root, self.kbest, tqdm_obj=tqdm_obj)
        
    def viterbi(self, conll_sentence, new_root_id, kbest, tqdm_obj=None):
        """
        Fixed tree decoder.
        What does it do as a black box?
        It modifies conll_sentence and returns (conll_sentence, score of best derivation)
        It sets the following attributes for each ConllEntry:
            - pred_edge_label
            - delex_supertag
            - typ
        :param conll_sentence: current sentence
        :param kbest: How many supertags to take into account? If not set, use the setting of the object (self.kbest)
        :return: the modified conll_sentence
        """
        chart = dict()
        backpointer = dict()
        agenda = []

        head_dict = get_head_dict(conll_sentence.heads[1:])
        t = tree.Tree.from_heads(conll_sentence.heads,conll_sentence)

        
        #~ print(t)
        #fill chart with complete items (leaves)
        max_arity = 0
        for sub_t in reversed(list(t.postorder())): 
            i,w = sub_t.node
            if w.typ == "ROOT-TYPE":
                continue
            if len(sub_t.children) > 0: #fill agenda
                max_arity = max(max_arity,sub_t.max_arity())
                a = []
                for item in self.get_items(w, kbest):
                    a.append(AgendaItem(i,head_dict[i],item.type,item.score,[(i,item.type)],[]))
                agenda.append(a)
            else: #fill chart
                chart[i] = dict()
                backpointer[i] = dict()
                for item in self.get_items(w, kbest):
                    chart[i][amtypes.typ_to_str(item.type)] = (item.type, item.score)
                    backpointer[i][amtypes.typ_to_str(item.type)] = ([(i, item.type)], [])
        #print()
        #print("Agenda",agenda)
        counter = 0

        cache = amtypes.CombinationCache()
        start_time = time()
        label_scores = conll_sentence.label_scores
        while agenda:
            one_index = agenda.pop()
            processed = dict()
            while one_index:
                if self.giveupat and time()-start_time > self.giveupat:
                    if kbest > 1:
                        msg = "Stop trying to parse with k={}, backing off to k={}".format(kbest,kbest-1)
                        if tqdm_obj:
                            tqdm_obj.write(msg)
                        else:
                            print(msg)
                        return self.viterbi(conll_sentence, new_root_id, kbest=kbest-1)
                    else:
                        if tqdm_obj:
                            tqdm_obj.write("Skipping sentence")
                        else:
                            print("Skipping sentence")
                        return conll_sentence
                it = one_index.pop()
                it_type_str = amtypes.typ_to_str(it.type)
                tupl = (it_type_str,frozenset(it.unproc_children))
                if tupl in processed and processed[tupl] >= it.score:
                    continue
                processed[tupl] = max(it.score,processed.get(tupl,it.score))
                todo = []

                for unprocessed in it.unproc_children:
                    for child_t, child_score in chart[unprocessed].values():
                        child_t_str = amtypes.typ_to_str(child_t)
                        for op,source in cache.combinations(it.type, it_type_str, child_t, child_t_str):
                            counter += 1
                            if op+source in self.rel2i:
                                edgescore = label_scores[unprocessed-1,self.rel2i[op+source]]
                                #edgescore = label_scores[self.rel2i[op+source], it.index, unprocessed]
                            else: #don't know that edge label => bad sign, but how bad?
                                edgescore = -100
                            #print(it.index, unprocessed,op+source,"mit Score",edgescore)
                            new_type = it.type
                            sub_bp = backpointer[unprocessed][child_t_str]
                            if op == "APP_": #type only changes for apply operations
                                new_type = amtypes.simulate_app(it.type, source)
                            new_it = AgendaItem(it.index, it.unproc_children - {unprocessed}, new_type, child_score + it.score + edgescore,it.local_types+sub_bp[0],sub_bp[1] + it.subtree + [(it.index, unprocessed,op+source)])

                            typ_str = amtypes.typ_to_str(new_it.type)
                            if len(new_it.unproc_children) == 0:
                                if new_it.index not in chart:
                                    chart[new_it.index] = dict()
                                    backpointer[new_it.index] = dict()
                                
                                if typ_str not in chart[new_it.index] or new_it.score > chart[new_it.index][typ_str][1]:
                                    chart[new_it.index][typ_str] = (new_it.type,new_it.score)
                                    backpointer[new_it.index][typ_str] = (new_it.local_types,new_it.subtree)
                            else:
                                #tupl = (typ_str,frozenset(new_it.unproc_children))
                                found = None
                                
                                for i,idx in enumerate(todo):
                                    if idx.unproc_children == new_it.unproc_children and idx.type == new_it.type:
                                        found = i
                                        break
                                if found:
                                    if new_it.score > todo[found].score: #fixed bug?
                                        todo[found] = new_it
                                else:
                                    todo.append(new_it)
                                        
                one_index.extend(sorted(todo, key=lambda it: it.score))
        for w in conll_sentence: #default: IGNORE
            w.pred_edge_label = "IGNORE"
            
        conll_sentence[new_root_id].pred_edge_label = "ROOT"
        #print(chart)
        #print("new root id", new_root_id)
        #print(head_dict)
        best_entry = max(chart[new_root_id].values(), key = lambda entry: (-amtypes.number_of_open_sources(entry[0]), entry[1])) #most important: few open sources, 2nd: highest score
        #Look up backpointers
        local_types,edges = backpointer[new_root_id][amtypes.typ_to_str(best_entry[0])]
        has_a_local_type = set()
        for index, typ in local_types:
            t_str = amtypes.typ_to_str(typ)
            has_a_local_type.add(index)
            conll_sentence[index].typ = t_str 
            for (score,delex,ty) in conll_sentence[index].supertags:
                if ty == t_str:
                    conll_sentence[index].delex_supertag = delex
                    break
        for (h,d,e) in edges:
            conll_sentence[d].pred_edge_label = e
        return conll_sentence
