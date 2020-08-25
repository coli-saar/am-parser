# cython: language_level=3
from typing import Set, Dict, Tuple, List, TypeVar, Generic, Iterable, Optional

NT = TypeVar('NT') # node type


class DiGraph: #Generic[NT]
    """
    Labeled directed (multi) graph, build with intention for being used in the types of the AM algebra.
    Nodes and edges are labeled by objects of the same type.
    """
    
    def __init__(self):
        self.origins : Set[NT] = set() #nodes without incoming edges
        self.edges : Dict[NT, Dict[NT, NT]] = dict() # edges[source][target] = label
        self._hash: Optional[int] = None
        
    def add_node(self, node : NT) -> None:
        self.origins.add(node)
        
        if node not in self.edges:
            self.edges[node] = dict()

        self._hash = None
        
    def add_edge(self, n1 : NT, n2: NT, label : NT) -> None:
        
        if n2 in self.origins:
            self.origins.remove(n2)
            
        self.edges[n1][n2] = label
        self._hash = None

    def get_parents(self, n : NT) -> Iterable[NT]:
        for some_node, children in self.edges.items():
            if n in children:
                yield some_node

    def __repr__(self) -> str:
        return "DiGraph<"+repr(self.origins)+","+repr(self.edges)+">"
    
    def closure(self):
        """
        Computes transitive closure.
        New edges get labeled with their target node.

        Returns
        -------
        None

        """
        self._hash = None
        for origin in self.origins:
            agenda : List[NT] = [ (node, {origin}) for node in self.edges[origin].keys()]
        
            while agenda:
                current_node, visited = agenda.pop()
                
                visited_new = visited | {current_node}
                
                for child in self.edges[current_node]:
                    agenda.append((child, visited_new))
                    
                for n in visited:
                    if current_node not in self.edges[n]:
                        self.add_edge(n, current_node, current_node)
                        

            
    def has_cycle(self) -> bool:
        #Reverse the dictionary for easier access below
        incoming_edges : Dict[NT, Set[NT]] = { o : set() for o in self.origins}
        for node in self.edges.keys():
            for tgt in self.edges[node]:
                if tgt not in incoming_edges:
                    incoming_edges[tgt] = set()
                incoming_edges[tgt].add(node)
                
        # Now, go over all nodes and delete those that have no incoming edges
        
        while True:
            remove = []
            
            for node in incoming_edges.keys():
                
                if len(incoming_edges[node]) == 0:
                    remove.append(node)
                    
                    for connected_to in self.edges[node]:
                        incoming_edges[connected_to].remove(node)
                        
            for node in remove:
                del incoming_edges[node]
                    
            if not incoming_edges:
                return False
            elif len(remove) == 0:
                return True
        
    def copy(self) -> "DiGraph":
        c = DiGraph[NT]()
        c.origins = set(self.origins)
        c.edges = { from_ : dict(self.edges[from_]) for from_ in self.edges.keys()}
        assert hash(c) == hash(self)

        return c
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, DiGraph):
            raise NotImplementedError("Comparison between DiGraph and "+str(type(other))+" not implemented.")
            
        return self.origins == other.origins and self.edges == other.edges
    
    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash
        self._hash = sum( (hash(node) % 90000000) * (1 + int(node in self.origins)) for node in self.edges)
        return self._hash
    
    def get_children(self, node : NT) -> Iterable[NT]:
        return self.edges[node]
    
    def is_empty(self):
        return len(self.edges) == 0
    
    def nodes(self) -> Iterable[NT]:
        return (node for node in self.edges)
    
    def outgoing_edges(self, node : NT) -> Iterable[Tuple[NT, NT]]:
        for n in self.edges[node]:
            yield (n, self.edges[node][n])
            
    def update_origins(self) -> None:
        self._hash = None
        incoming_edges : Set[NT] = set()
        for n1 in self.edges:
            for n2 in self.edges[n1]:
                incoming_edges.add(n2)
        self.origins = set(self.edges.keys()) - incoming_edges
            
    def remove_node(self, node : NT) -> None:
        self._hash = None
        del self.edges[node]
        
        if node in self.origins:
            self.origins.remove(node)
        else:
            #node has incoming edges
            for n in self.edges:
                if node in self.edges[n]:
                    del self.edges[n][node]
                    
        #children might become origins
        self.update_origins()
        
        
         
           
            
        
        

if __name__ == "__main__":
    import timeit
    
    g = DiGraph[str]()
    
    g.add_node("a")
    g.add_node("b")
    g.add_node("c")
    g.add_node("d")

    g.add_edge("a","b","X")
    g.add_edge("b","c","Y")
    g.add_edge("d","b","U")
    
    
    
    print(g)
    g.closure()
    print(g)
    print(g.has_cycle())
    