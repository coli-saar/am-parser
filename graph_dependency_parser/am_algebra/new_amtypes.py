#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:47:47 2019

@author: matthias
"""

from networkx import DiGraph
import networkx.algorithms.dag as algs

from typing import Set, Dict, Tuple, List, Iterator, Optional
import re
from copy import deepcopy




def tokenize(type_str) -> List[str]:
    return list(filter(lambda x: x!="" and x!=None, re.split("([(),])| ",type_str)))

def extract_constituents(depths) -> List[Tuple[int,int]]:
    """
    Given a list of depths, return a list of tuples that are directly subordinate
    that is, identify chunks of the form 0 1 * 0
    """
    start = 0
    ret : List[Tuple[int,int]] = []
    for i,d in enumerate(depths):
        if d == 0 and i+1 < len(depths) and depths[i+1] == 1:
            start = i+1
        elif d == 0 and i > 1 and depths[i-1] == 1:
            ret.append((start,i-1))
    return ret
            

UNIFY_PATTERN : re.Pattern = re.compile("(.+)_UNIFY_(.+)")
UNIFY = "_UNIFY_"

class AMType:
    
    def __init__(self):
        self.graph = DiGraph()
        self.is_bot : bool = False #\bot type?
        self.origins = set()
        
    def update_origins(self):
         self.origins = { node for node, indegree in self.graph.in_degree(self.graph.nodes) if indegree == 0}
         
    def ensure_closure(self):
        self.graph = algs.transitive_closure(self.graph)
        
    def ensure_edge_labels(self):
        for f,t in self.graph.edges():
            if "label" not in self.graph[f][t]:
                self.graph[f][t]["label"] = t
            
    def process_updates(self):
        self.update_origins()
        self.ensure_closure()
        self.ensure_edge_labels()
        
        assert algs.is_directed_acyclic_graph(self.graph)
        
        assert self.verify()
            
        
    def outgoing_edges(self, node) -> Iterator[Tuple[str,str]]:
        for target, rename in self.graph[node].items():
            yield target, rename["label"]
            
    def get_children(self, node) -> Set[str]:
        return set(self.graph[node])
        
    def verify(self) -> bool:
        """
        Checks that all edges leaving a node are uniquely labeled. See definition 5.2. (iv)
        """
        if self.is_bot:
            if len(self.graph.nodes()) > 0:
                return False

        for node in self.graph.nodes():
            seen : Set[str] = set()
            for _, label in self.outgoing_edges(node):
                if label in seen:
                    return False
                else:
                    seen.add(label)
        return True
        

    # string to type
        
    @staticmethod 
    def parse_str(s : str) -> "AMType":
        tokens = tokenize(s)
        
        if tokens == ["_"] or tokens == ["NULL"]:
            t = AMType()
            t.is_bot = True
            return t
        
        assert tokens[-1] == ")"
        
        t = AMType._parse_tokens(tokens[:-1])
        
        t.process_updates()
        
        return t
        
    @staticmethod
    def _parse_tokens(s : List[str], parent = None, typ : "AMType" = None) -> "AMType":
        assert s[0] == "("
        s = s[1:]
        depth = 0
        depths = []
        parents = []
        if typ is None:
            typ = AMType()
        for t in s:
            if t == "(":
                depth += 1
            elif t == ")":
                depth -= 1
            elif t == ",":
                pass
            elif depth == 0:
                m = UNIFY_PATTERN.fullmatch(t)
                if m: #rename, e.g. o2(s_UNIFY_o)
                    typ.graph.add_node(m.group(2))
                    assert parent is not None
                    parents.append(m.group(2))
                    typ.graph.add_edge(parent, m.group(2),label=m.group(1))
                else:
                    typ.graph.add_node(t)
                    parents.append(t)
                    if parent is not None:
                        typ.graph.add_edge(parent, t, label=t)
            depths.append(depth)
            
        subexprs = [ s[from_:to_+1] for (from_, to_) in extract_constituents(depths)]

        for subexpr,parent in zip(subexprs, parents):
            AMType._parse_tokens(subexpr,parent,typ)
                    
        return typ
    
    # type to string
    
    def _dominated_subgraph_str(self, node : str) -> str:
        r = []
        for target, label in sorted(self.outgoing_edges(node), key=lambda x:x[0]):
            eRep = ""
            if label != target:
                eRep = label+UNIFY
            r.append(eRep + self._dominated_subgraph_str(target))
              
        return node+"(" + ", ".join(r) + ")"
    
    def __str__(self):
        if self.is_bot:
            assert self.verify()
            return "_"
        
        r = []
        for o in sorted(self.origins):
            r.append(self._dominated_subgraph_str(o))
        return "("+", ".join(r)+")"
    
    def __repr__(self):
        return str(self)
    
    def _depth(self) -> int:
        """
        [] has depth 0, [O] has depth 1, [O[S]] has depth 2, etc.
        """
        raise NotImplementedError()
        
        
    def is_compatible_with(self, other : "AMType") -> bool:
        """
        Checks whether this type is a subgraph of type 'other'.
        """
        node_subset = set(self.graph.nodes()).issubset(other.graph.nodes())
        if not node_subset:
            return False
        
        for node in self.graph.nodes():
            for t, label in self.outgoing_edges(node):
                if t not in other.graph[node]:
                    return False # edge present in self but not in other
                elif other.graph[node][t]["label"] != label:
                    return False # edge present in both but not the same label
                
        return True


    def to_request_namespace(self, parent, child) -> Optional[str]:
        """
        Maps the descendant/child to its counterpart in req(parent). Returns None
        if "descendant" is not actually a descendant of parent.
        """
        if parent not in self.graph.nodes() or child not in self.graph.nodes() or child not in self.graph[parent]:
            return None
        return self.graph[parent][child]["label"]

    def get_request(self, source : str) -> Optional["AMType"]:
        """
        Returns the request of this type at source s (=req(s)).
        Returns None if s is not present in self
        """
        if source not in self.graph:
            return None
        
        descendants = self.get_children(source)
        ret = AMType()
        
        for node in descendants:
            ret.graph.add_node(self.to_request_namespace(source,node))
            
        for node in descendants:
            for target, label in self.outgoing_edges(node):
                assert target in descendants, "Type seems to be NOT transitively closed"
                
                ret.graph.add_edge(self.to_request_namespace(source, node),
                                   self.to_request_namespace(source, target),
                                   label=label
                                   )
                    
        ret.process_updates()
        return ret
    
    def is_empty_type(self) -> bool:
        """
        Is this the empty type?
        """
        return len(self.graph.nodes()) == 0 and not self.is_bot
    
    
    def get_apply_set(self, target : "AMType") -> Optional[Set[str]]:
        """
        Returns the set of source names such that if we call apply for all those
        source names on this type (self), the given type target remains. Returns None if
        no such list of source names exists.
        """
        if self.is_bot:
            if target.is_bot:
                return set()
            return None
        
        if not isinstance(target, AMType) or not target.is_compatible_with(self):
            return None
        
        #return value is all sources in this type that are not in target
        ret = set(self.graph.nodes()) - set(target.graph.nodes())
        
        
        # but if any source s in ret is a descendant of a node t in target,
        # then we can't remove s via apply without removing t before.
        # Can check for that by just looking at the children of the nodes in target.
        
        for node in target.graph.nodes():
            if self.get_children(node) & ret:
                return None
            
        return ret
    
    def can_apply_to(self, argument : "AMType", source : str) -> bool:
        """
         Checks whether APP_appSource(G_1, G_2) is allowed, given G_1 has this
         type, and G_2 has type 'argument'.
        """
        if self.is_bot: #to make it explicit: \bot cannot be used in APP
            return False
        
        request = self.get_request(source)
        # check if the type expected here at source is equal to the argument type
        if request is None or request != argument:
            return False
        # check that appSource is an origin (and thus not needed for unification later)
        return source in self.origins
    
    def can_be_modified_by(self, modifier: "AMType", source : str) -> bool:
        """
        Checks whether MOD_modSource(G_1, G_2) is allowed, given G_1 has this
         * type, and G_2 has type argument.
        """
        if self.is_bot: # \bot cannot be modified
            return False
        
        request = modifier.get_request(source)
        
        return request is not None and \
            request.is_empty_type() and \
            source in modifier.origins and \
            modifier.copy_with_removed(source).is_compatible_with(self)
            
    def copy_with_removed(self, source : str) -> "AMType":
        """
         Creates a copy with r removed from the domain. Does not modify the
         original type.
        """
        copy = deepcopy(self)
        copy.graph.remove_node(source)
        copy.process_updates()
        return copy
    
    def can_apply_now(self, source : str) -> bool:
        """
        Are we currently allowed to apply at this source?
        (note: \bot has no)
        """
        return source in self.origins
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, AMType):
           raise NotImplementedError("Comparison between AMType and "+str(type(other))+" not implemented.")
        
        if self.origins != other.origins:
            return False
        
        if self.is_bot != other.is_bot:
            return False
        
        if set(self.graph.nodes()) != set(other.graph.nodes()):
            return False
        
        for node in self.graph.nodes():
            if set(self.outgoing_edges(node)) != set(other.outgoing_edges(node)):
                return False
        
        return True
    
    def __hash__(self) -> int:
        h = sum(hash(s) % 999999 for s in self.origins)
        
        for node in self.graph.nodes():
            for o,label in self.outgoing_edges(node):
                h += (hash(node) % 100000)  * (hash(o) % 500000 ) * (hash(label) % 300000)
                
        return h

    def perform_apply(self, source : str) -> Optional["AMType"]:
        """
         * Returns the type that we obtain after using APP_s on this type (returns
         * a copy and does
         * not modify this type). Returns null if APP_s is not allowed for this
         * type (i.e. if this type does not contain s, or s is needed for later
         * unification).
         *
         * This is a bit redundant with performOperation and copyWithRemoved,
         * but performOperation is clunkier to use (and requires righthand side type)
         * and copyWithRemoved is not public. So I think this has a place -- JG
         * 
         """
        if not self.can_apply_now(source):
             return None
         
        return self.copy_with_removed(source)
    
    def number_of_open_sources(self) -> float:
        """
        Returns the number of sources left or infinity if this is the \bot type.
        """
        if self.is_bot:
            return float("inf")
        
        return len(self.graph.nodes())
    
        

def combinations(head : AMType, dependent : AMType) -> Iterator[Tuple[str,str]]:
    """
    return the operations allowed between head and dependent.
    The first value of the return value is the operation (APP, MOD, IGNORE)
    the second one is the source
    """
    if dependent.is_bot:
        return ("IGNORE","")
    
    for o in head.origins:
        if head.can_apply_to(dependent, o):
            yield ("APP_",o)
            
    for o in dependent.origins:
        if head.can_be_modified_by(dependent, o):
            yield ("MOD_",o)

class CombinationCache:
    
    def __init__(self):
        self.cache = dict()
        
    def combinations(self,head,dependent) -> Set[Tuple[str,str]]:
        try:
            return self.cache[(head,dependent)]
        except KeyError:
            combis = set(combinations(head,dependent))
            self.cache[(head,dependent)] = combis
            return combis


if __name__ == "__main__":
    from tqdm import tqdm
    t = AMType.parse_str("(s(), o()")
    t2 = AMType.parse_str("()")
    bot = AMType.parse_str("_")
    print("===")
    print(t)
    
    print(t.get_apply_set(AMType.parse_str("(o2)")))
    
    #TODO: (s(mod_UNIFY_o(s_UNIFY_o2()), o2())) wird nicht geparst
    
    if False:
    
        with open("type_judgements.txt") as f:
                for line in tqdm(f):
                    t1,t2,op,s,ok = line.strip().split("\t")
                    try:
                        t1 = AMType.parse_str(t1)
                    except Exception as e:
                        print("Fehler",e,t1)
                    try:
                        t2 = AMType.parse_str(t2)
                    except Exception as e:
                        print("Fehler",e,t2)
                        
                    if op == "APP" and isinstance(t1, AMType) and isinstance(t2, AMType):
                        answer = str(t1.can_apply_to(t2,s)).lower()
                        if answer != ok:
                            print("Anders",t1,t2,op,s,ok)
                    elif op == "MOD" and isinstance(t1, AMType) and isinstance(t2, AMType):
                        answer = str(t1.can_be_modified_by(t2,s)).lower()
                        if answer != ok:
                            print("Anders",t1,t2,op,s,ok)
                    else:
                        print("???",op, t1,t2)
 