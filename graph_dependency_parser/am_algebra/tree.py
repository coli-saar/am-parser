
class Tree:
    """
    A simple tree class
    """
    def __init__(self, node, children):
        self.node = node
        self.children = children
        
    def add_child(self,child):
        self.children.append(child)
    
    @staticmethod
    def from_heads(heads, conll_sentence):
        
        def parse(i):
            mother = Tree((i,conll_sentence[i]),[])
            for j in range(len(heads)):
                if heads[j] == i:
                    mother.add_child(parse(j))
            return mother
        
        return parse(0) #articial root is at position 0
        
    
    def fold(self, f):
        """
        Folding on trees: f takes a node a list of things that f produces
        Returns a single thing (for instance a Tree or a number)
        """
        if len(self.children) == 0:
            return f(self.node,[])
        return f(self.node,[c.fold(f) for c in self.children])

    def fold_double(self, f):
        """
        Folding on trees: f takes a node a list of things that f produces
        Returns a single thing (for instance a Tree or a number)
        """
        if len(self.children) == 0:
            return f(self.node,[],[])
        return f(self.node,self.children,[c.fold_double(f) for c in self.children])
        
    def map(self,f):
        """
            applies f to all nodes of the tree. changes the tree
        """
        self.node = f(self.node)
        for c in self.children:
            c.map(f)
    
    def size(self):
        if len(self.children) == 0:
            return 1
        return 1+sum(c.size() for c in self.children)

    def max_arity(self):
        if len(self.children) == 0:
            return 0
        return max(len(self.children), max(c.max_arity() for c in self.children))
        
    def postorder(self):
        if self.children == []:
            yield self
        else:
            for c in self.children:
                for x in c.postorder():
                    yield x
            yield self

    def _to_str(self,depth=0):
        if len(self.children) == 0:
            return 4*depth*" "+str(self.node)
        return 3*depth*" "+"["+str(self.node)+"\n {}]".format("\n".join( c._to_str(depth+1) for c in self.children))
    
    def __str__(self):
        return self._to_str()

    def __repr__(self):
        if len(self.children) == 0:
            return "("+str(self.node) +")"
        return "({} {})".format(str(self.node)," ".join(c.__repr__() for c in self.children))


if __name__ == "__main__":
    t = Tree("a",[Tree("b",[]),Tree("c",[Tree("d",[])])])

    #~ h = [-1, 2, 29, 8, 8, 2, 2, 8, 0, 5, 5, 14, 14, 14, 21, 17, 17, 18, 14, 14, 29, 22, 20, 22, 22, 29, 27, 29, 29, 4]
    #~ t = Tree.from_heads(list(h),range(len(h)))
    #~ print(t)
    #print(t.fold(lambda n,children: Tree(n[0],children)))
    
    #t.map(lambda node: node[0]) #the same thing as folding, but with side-effect
    #print(t)
    print(t.size())
    print(list(t.postorder()))
