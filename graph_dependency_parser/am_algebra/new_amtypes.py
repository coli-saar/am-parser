#!/usr/bin/env python3
# cython: language_level=3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:47:47 2019

@author: matthias
"""

from graph_dependency_parser.am_algebra.dag import DiGraph

from typing import Set, Dict, Tuple, List, Iterator, Optional
import re



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

class AMType(DiGraph[str]):
    
    def __init__(self):
        super().__init__()
        self.is_bot : bool = False #\bot type?
        
            
    def process_updates(self):
        self.closure()
        
        assert not self.has_cycle()
        
        assert self.verify()
            
        
    def verify(self) -> bool:
        """
        Checks that all edges leaving a node are uniquely labeled. See definition 5.2. (iv)
        """
        if self.is_bot:
            if not self.is_empty():
                return False

        for node in self.nodes():
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
                    typ.add_node(m.group(2))
                    assert parent is not None
                    parents.append(m.group(2))
                    typ.add_edge(parent, m.group(2),label=m.group(1))
                else:
                    typ.add_node(t)
                    parents.append(t)
                    if parent is not None:
                        typ.add_edge(parent, t, label=t)
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
        node_subset = set(self.nodes()).issubset(other.nodes())
        if not node_subset:
            return False
        
        for node in self.nodes():
            for t, label in self.outgoing_edges(node):
                if t not in other.edges[node]:
                    return False # edge present in self but not in other
                elif other.edges[node][t] != label:
                    return False # edge present in both but not the same label
                
        return True


    def to_request_namespace(self, parent, child) -> Optional[str]:
        """
        Maps the descendant/child to its counterpart in req(parent). Returns None
        if "descendant" is not actually a descendant of parent.
        """
        if parent not in self.nodes() or child not in self.nodes() or child not in self.edges[parent]:
            return None
        return self.edges[parent][child]

    def get_request(self, source : str) -> Optional["AMType"]:
        """
        Returns the request of this type at source s (=req(s)).
        Returns None if s is not present in self
        """
        if source not in self.edges:
            return None
        
        descendants = self.get_children(source)
        ret = AMType()
        
        for node in descendants:
            in_namespace = self.to_request_namespace(source,node)
            if in_namespace is not None:
                ret.add_node(in_namespace)
            else:
                return None
            
        for node in descendants:
            for target, label in self.outgoing_edges(node):
                assert target in descendants, "Type seems to be NOT transitively closed"
                
                source_r = self.to_request_namespace(source, node)
                target_r = self.to_request_namespace(source, target)
                if source_r is None or target_r is None:
                    return None
                ret.add_edge(source_r,target_r,label=label)
                
                    
        ret.process_updates()
        return ret
    
    def is_empty_type(self) -> bool:
        """
        Is this the empty type?
        """
        return self.is_empty() and not self.is_bot
    
    
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
        ret = set(self.nodes()) - set(target.nodes())
        
        
        # but if any source s in ret is a descendant of a node t in target,
        # then we can't remove s via apply without removing t before.
        # Can check for that by just looking at the children of the nodes in target.
        
        for node in target.nodes():
            if set(self.get_children(node)) & ret:
                return None
            
        return ret
    
    def can_apply_to(self, argument : "AMType", source : str) -> bool:
        """
         Checks whether APP_appSource(G_1, G_2) is allowed, given G_1 has this
         type, and G_2 has type 'argument'.
        """
        if self.is_bot: #to make it explicit: \bot cannot be used in APP
            return False

        # check that appSource is an origin (and thus not needed for unification later)
        if source not in self.origins:
            return False

        request = self.get_request(source)
        # check if the type expected here at source is equal to the argument type

        return request is not None and request == argument
    
    def can_be_modified_by(self, modifier: "AMType", source : str) -> bool:
        """
        Checks whether MOD_modSource(G_1, G_2) is allowed, given G_1 has this
         * type, and G_2 has type argument.
        """
        if self.is_bot: # \bot cannot be modified
            return False

        if source not in modifier.origins:
            return False
        
        request = modifier.get_request(source)
        
        return request is not None and \
            request.is_empty_type() and \
            modifier.copy_with_removed(source).is_compatible_with(self)
            
    def copy_with_removed(self, source : str) -> "AMType":
        """
         Creates a copy with r removed from the domain. Does not modify the
         original type.
        """
        copy = self.copy()
        copy.remove_node(source)
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
        
        if self.is_bot != other.is_bot:
            return False
        
        return super().__eq__(other)
        

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
        
        return sum(1 for _ in self.nodes())
    
    def copy(self) -> "AMType":
        g = AMType()
        g.origins = set(self.origins)
        g.edges = { from_ : dict(self.edges[from_]) for from_ in self.edges.keys()}
        g.is_bot = self.is_bot
        
        return g
    
    def __hash__(self) -> int:
        return sum((hash(node) % 90000000) * (1 + int(node in self.origins)) for node in self.edges)

        

def combinations(head : AMType, dependent : AMType) -> Iterator[Tuple[str,str]]:
    """
    return the operations allowed between head and dependent.
    The first value of the return value is the operation (APP, MOD, IGNORE)
    the second one is the source
    """
    if dependent.is_bot:
        yield ("IGNORE","")
        return

    if head.is_bot:
        return
    
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
    import timeit
    t = AMType.parse_str("(s(), o()")
    t2 = t.copy()
    print(timeit.timeit(lambda: t.copy(),number = 40000))
    t3 = AMType.parse_str("()")
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
 