# cython: language_level=3

from operator import itemgetter
from typing import Dict, Tuple

import graph_dependency_parser.am_algebra.tree as tree
import tqdm
from graph_dependency_parser.am_algebra.amconll import ConllEntry, ConllSent, write_conll
from graph_dependency_parser.components.cle import get_head_dict

from time import time

import pyximport; pyximport.install()
#import graph_dependency_parser.am_algebra.amtypes as amtypes
import graph_dependency_parser.am_algebra.new_amtypes as new_amtypes

#import graph_dependency_parser.am_algebra.evaluator as evaluator
import multiprocessing as mp
import numpy as np

class AgendaItem:
    """
    Represents a parse item that lives on the agenda.
    """
    def __init__(self,index,unproc_children,type,score,local_types,subtree, is_art_root):
        self.index = index
        self.unproc_children = unproc_children
        self.type = type
        self.score = score
        self.local_types = local_types
        self.subtree = subtree
        self.is_art_root = is_art_root

        self.already_applied = False #did we use apply already?

    def __str__(self):
        return "AgendaItem({},{},{},{},{},{},{})".format(self.index,self.unproc_children,self.type,self.score,self.local_types,self.subtree, self.already_applied)
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
        print("Using new type implementation")
        self.totyp = dict() #look up table to avoid parsing the same type from string to python object twice
        self.apply_cache = dict() # Dict[Tuple[new_amtypes.AMType, str], new_amtypes.AMType]

        self.output_file = output_file
        self.i2rel = i2edgelabel
        self.rel2i = { rel : i for i, rel in enumerate(i2edgelabel)}

        self.sents = []
    

    def add_sentence(self, root : int, heads, label_scores, lexlabels, supertagpreds, sentence, attributes):
        """
        Add a new sentence to our todo list of sentences to parse.
        In the following, let N be the length of the sentence, L the number of different labels
        :param root: index of the root of dependency tree, 1-based
        :param heads: a numpy array of length N, the head of every word.
        :param label_scores: a numpy array for edge label scores with dimensions (N+1) x L. label_scores[i,l] represents the score for the incoming edge to i having label l
        :param lexlabels: a list of of lexical labels (strings) that will be used for the words
        :param supertagpreds: a list of lists. For each word, this list contains triples (log probability, graph fragment as string, AM type as string) and must contain an entry for \bot (represented by _)
        :param sentence: a list of tuples in the shape: (form,replacement,lemma,pos,ne, token_range)
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
        for i,(form,replacement,lemma,pos,ne, token_range) in enumerate(sentence):
            entry = ConllEntry(i+1,form,replacement,lemma,pos,ne,"_",lexlabels[i],"_",0,"IGNORE",True,token_range)
            entry.pred_parent_id = heads[i]
            if not any(triple[2] == "_" for triple in supertagpreds[i]):
                raise ValueError("The supertag prediction for word "+str(i+1)+" has no entry for bottom (represented by _). This is required.")
            entry.supertags = sorted(supertagpreds[i], key= lambda triple: triple[0], reverse=True)
            conllsent.append(entry)
        assert len(conllsent) == len(heads)+1

        self.sents.append((conllsent,))


    def decode(self, threads, kbest, giveup, give_up_k_1=None):
        """
        Actually parse the sentences. Write the results to the output file.
        :param threads:
        :param kbest:
        :param giveup: time limit for each sentence until we back off to lower k
        :param give_up_k_1: time limit for each sentence until we stop trying to parse with k=1 and skip it
        :return:
        """
        self.kbest = kbest
        self.giveupat = giveup

        if give_up_k_1 is None:
            give_up_k_1 = giveup
        self.give_up_k_1 = give_up_k_1

        if self.giveupat:
            print("Will give up after",self.giveupat,"s when still not successful")
        print("Using",kbest,"best supertags")
        t1 = time()
        if threads > 1:
            with mp.Pool(threads) as pool:
                r = list(pool.starmap(self.call_viterbi,self.sents))
        else:
            tqdm_sents = tqdm.tqdm(self.sents)
            r = [ self.call_viterbi(s[0], tqdm_obj=tqdm_sents) for s in tqdm_sents]
        t2 = time()
        print("Parsing with fixed-tree decoder took",round(t2-t1,3),"seconds.")
        write_conll(self.output_file,r)
        self.sents = []










        
    def parse_am_type(self, string):
        
        if string in self.totyp:
            return self.totyp[string]
        self.totyp[string] = new_amtypes.AMType.parse_str(string)
        return self.totyp[string]

    def perform_apply(self, typ : new_amtypes.AMType, source : str) -> new_amtypes.AMType:
        tupl = (typ, source)
        if tupl in self.apply_cache:
            return self.apply_cache[tupl]
        self.apply_cache[tupl] = typ.perform_apply(source)
        return self.apply_cache[tupl]

    def get_items(self, entry, kbest):
        """
            Creates a list of Items for a (leaf) entry. Assumes entry to have a list called supertags where the tags are sorted in descending order by score.
        """
        ms = []
        types_used = []
        null_item = None
        is_art_root = entry.form == "ART-ROOT"
        bot_type = self.parse_am_type("_")
        for (s,delex,typ) in entry.supertags:
            t = self.parse_am_type(typ)
            if not t in types_used:
                ms.append(AgendaItem(entry.id,set(),t,s,[(entry.id,t)],[],is_art_root))
                types_used.append(t)
            if t == bot_type:
                null_item = AgendaItem(entry.id,set(),t,s,[(entry.id,t)],[],is_art_root)
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
        chart : Dict[int,Dict[new_amtypes.AMType,Tuple[new_amtypes.AMType, float]]] = dict()
        backpointer = dict()
        agenda = []

        head_dict = get_head_dict(conll_sentence.heads[1:])
        t = tree.Tree.from_heads(conll_sentence.heads,conll_sentence)

        
        #~ print(t)
        #fill chart with complete items (leaves)
        encountered_root = False
        max_arity = 0
        for sub_t in reversed(list(t.postorder())): 
            i,w = sub_t.node
            if i == new_root_id:
                encountered_root = True
            if w.typ == "ROOT-TYPE":
                continue
            if len(sub_t.children) > 0: #fill agenda
                max_arity = max(max_arity,sub_t.max_arity())
                a = []
                for item in self.get_items(w, kbest):
                    a.append(AgendaItem(i,head_dict[i],item.type,item.score,[(i,item.type)],[],w.form == "ART-ROOT"))
                agenda.append(a)
            else: #fill chart
                chart[i] = dict()
                backpointer[i] = dict()
                for item in self.get_items(w, kbest):
                    chart[i][item.type] = (item.type, item.score)
                    backpointer[i][item.type] = ([(i, item.type)], [])
        if not encountered_root:
            print("Didn't encounter root when constructing the agenda, something's wrong :( ","root",new_root_id)
            print("tree",head_dict,t)
            print("sentence",  str(conll_sentence))
        counter = 0

        cache = new_amtypes.CombinationCache()
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
                    elif time()-start_time > self.give_up_k_1:
                        if tqdm_obj:
                            tqdm_obj.write("Skipping sentence")
                        else:
                            print("Skipping sentence")
                        return conll_sentence
                it = one_index.pop()
                tupl = (it.type,frozenset(it.unproc_children))
                if tupl in processed and processed[tupl] >= it.score:
                    continue
                processed[tupl] = max(it.score,processed.get(tupl,it.score))
                todo = []

                for unprocessed in it.unproc_children:
                    for child_t, child_score in chart[unprocessed].values():
                        for op,source in cache.combinations(it.type, child_t):
                            if op == "MOD_":
                                if it.is_art_root: #at ART-ROOT: only allow APP operations with sources starting with art-snt
                                    continue
                                elif it.already_applied: #we don't have to try that because we can first perform all MOD than all APP
                                    continue

                            counter += 1
                            if op+source in self.rel2i:
                                edgescore = label_scores[unprocessed-1,self.rel2i[op+source]]
                                #edgescore = label_scores[self.rel2i[op+source], it.index, unprocessed]
                            else: #don't know that edge label => bad sign, but how bad?
                                edgescore = -100
                            #print(it.index, unprocessed,op+source,"mit Score",edgescore)
                            new_type = it.type
                            sub_bp = backpointer[unprocessed][child_t]
                            op_is_APP = op == "APP_"

                            if op_is_APP: #type only changes for apply operations
                                new_type = self.perform_apply(it.type, source) #it.type.perform_apply(source)
                                assert new_type is not None, "applying seems disallowed although the operation came from the CombinationCache"

                            new_it = AgendaItem(it.index, it.unproc_children - {unprocessed}, new_type, child_score + it.score + edgescore,it.local_types+sub_bp[0],sub_bp[1] + it.subtree + [(it.index, unprocessed,op+source)],it.is_art_root)

                            if op_is_APP:
                                new_it.already_applied = True
                            else:
                                new_it.already_applied = it.already_applied

                            if len(new_it.unproc_children) == 0:
                                if new_it.index not in chart:
                                    chart[new_it.index] = dict()
                                    backpointer[new_it.index] = dict()
                                
                                if new_type not in chart[new_it.index] or new_it.score > chart[new_it.index][new_type][1]:
                                    chart[new_it.index][new_type] = (new_it.type,new_it.score)
                                    backpointer[new_it.index][new_type] = (new_it.local_types,new_it.subtree)
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
        try:
            best_entry = max(chart[new_root_id].values(), key = lambda entry: (-(entry[0].number_of_open_sources()), entry[1])) #most important: few open sources, 2nd: highest score
        except KeyError as e:
            print(e)
            print("Didn't find an item for the root of the AM dependency tree. This must not happen :/")
            print(head_dict)
            print(t)
            print("\n".join([str(e) for e in conll_sentence]))
            print([self.get_items(entry, kbest) for entry in conll_sentence[1:]])
            print("Skipping sentence")
            return conll_sentence
        #Look up backpointers
        local_types,edges = backpointer[new_root_id][best_entry[0]]
        has_a_local_type = set()

        for index, typ in local_types:
            t_str = str(typ)
            has_a_local_type.add(index)
            conll_sentence[index].typ = t_str

            for (score,delex,ty) in conll_sentence[index].supertags:
                if self.parse_am_type(ty) == typ:
                    conll_sentence[index].delex_supertag = delex
                    break

        for (h,d,e) in edges:
            conll_sentence[d].pred_edge_label = e
        return conll_sentence
