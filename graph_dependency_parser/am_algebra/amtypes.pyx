# cython: language_level=3
import re


####----------------------------------------- PARSING AM TYPES ----------------------------------------------

# S -> ()
# S -> _
# S -> ( X )
# X -> Name S
# X -> X , X


def X(name,s): #semantics of category X
    if "_UNIFY_" in name:
        t1,t2 = name.split("_UNIFY_")
        return { t1: (t2,s)}
    return {name:s}
    
rules = {
("\(","\)") : ("S",lambda x,y: None),
("\(","X","\)") : ("S",lambda l,x,r: x),
#("[A-Za-z0-9_]+","S") : ("X",lambda name,s: {name:s}),
("[-A-Za-z0-9_]+","S") : ("X",X),
#("[A-Za-z0-9]+","S",",","X") : "X",
("X",",","X") : ("X",lambda x1,k,x2:dict_merge(x1,x2)),
("_",) : ("S",lambda x:False), #not a type
}

def dict_merge(d1,d2):
    temp = dict(d1)
    temp.update(d2)
    return temp

def tokenize(type_str):
    return list(filter(lambda x: x!="" and x!=None, re.split("([(),])| ",type_str)))
    
def check_rule(stack):
    """
    Checks whether a syntax rule is applicable to the top of the stack.
    """
    for r in rules:
        if len(r) <= len(stack):
            stack_slice = stack[-len(r):]
            #print(stack,r)
            for tok,slice in zip(r,stack_slice):
                if not re.fullmatch(tok,slice):
                    break
            else:
                #print("passt")
                return (rules[r],len(r))
    return None
            
def shift_reduce(tokens):
    """
    Parses a sequence of tokens into a type
    """
    stack = []
    val_stack = []
    tokens = list(reversed(tokens))
    while tokens or len(stack)>0:
        #print("check_rule")
        s = check_rule(stack)
        if len(stack) == 1 and stack[0] == "S" and len(tokens) == 0:
            return val_stack[0]
        if s: #reduce
            try:
                (ersetzung,f), l = s
                stack[-l:] = [ersetzung]

                val_stack[-l:] = [f(*val_stack[-l:])] #apply corresponding semantics

            except Exception:
                raise ValueError("type parse error. stack:"+stack+" val_stack:"+val_stack )
        else: #shift
            if len(tokens) == 0:
                raise ValueError("type parse error: buffer empty")
            t = tokens.pop()
            stack.append(t)
            val_stack.append(t)
    return val_stack[0]
    
def parse_type(string):
    return shift_reduce(tokenize(string))


### ------------------------------------------------------ UTILITIES -----------------------------------------------------
def get_all_unif_targets(typ):
    """
    Returns the set of all results of renaming operations (recursively)
    """
    if not typ:
        return set()
    r = set()
    for source in typ:
        if isinstance(typ[source],tuple):
            r.add(typ[source][0])
        elif isinstance(typ[source],dict):
            r.update(get_all_unif_targets(typ[source]))
    return r

def extract_expectation(d):
    """
    Strips of tuples that contain renaming information
    """
    if not isinstance(d,dict):
        return d
    r = []
    for source, e in d.items():
        if isinstance(e,tuple):
            rename,e = e
        r.append((source,extract_expectation(e)))
    return dict(r)
    
    
def _subset(head, mod):
    if mod == None:
        return True
    if head == None: #head == None but mod != None
        return False
    for k in mod.keys():
        if mod[k] == None: #when the modifier expects None that's always fine (subset of everything)
            continue
        #print(head,k)
        if k not in head:
            return False
        source,rename = mod[k]
        if not source in head[k]:
            return False
        if not _subset(head[k][1],mod[k][1]):
            return False
        
        
    return True


def get_symbols(typ):
    """
    Retrieves all sources from Type typ (ignores renaming)
    """
    s = set()
    typ = extract_expectation(typ) #remove renaming
    if not typ:
        return s
    s.update(typ.keys())
    for subt in typ:
        s.update(get_symbols(typ[subt]))
    return s

def typ_to_str(typ):
    """
    Creates the Alto string representation of a type
    """
    if typ == False:
        return "_"
    elif typ == None:
        return "()"
    elif isinstance(typ,tuple):
        return "_UNIFY_{}{}".format(typ[0],typ_to_str(typ[1]))
    else:
        return "("+", ".join([ k+typ_to_str(typ[k]) for k in sorted(typ.keys())])+")"

    

###--------------------------------------------------------------- CORE CONTENT -----------------------------------------------------
    
def is_app_applicable(head,complement,alpha):
    """
    APP_alpha(head,complement) 
    """
    if not head: #head is either None or False, both are not compatible with anything using apply
        return False
    if not alpha in head:
        return False
    targets = get_all_unif_targets(head)
    if alpha in targets: #we can't apply yet, because something wants to rename a source to alpha but is not applied yet
        return False
        
    return extract_expectation(head[alpha]) == extract_expectation(complement) 
    
def simulate_app(head,source):
    """
    Returns the type that we get after filling source in head. Use only, when there is something that can be applied to head using source
    """
    if not source in head:
        raise ValueError("Typ "+typ_to_str(head)+" has no source "+source)
    expectation = head[source]
    r = dict(head)
    del r[source]
    if expectation != None:
        for (rename,expects) in expectation.values():
            if rename not in r: #only add a new source to the top-level if it didn't exist before
                r[rename] = expects
    
    if len(r) == 0: #instead of the empty dictionary, we use None
        return None
    return r
        

    
def is_mod_applicable(head,modifier,alpha):
    if not modifier: #modifier must have a source
        return False
    if head==False:
        return False
    if not alpha in modifier:
        return False
    if not modifier[alpha] == None: #no complex expectations of modifier source
        return False
    if head == None:
        head_keys = set()
    else:
        head_keys= head.keys()
    keys = set(modifier.keys()) - {alpha}
    if not keys.issubset(head_keys): #the modifier must not introduce a new source
        return False
    # make sure that the sources of the modifier have at most the same expectations (including renaming) as the head (if modifier has None expectation for something, that's always fine)
    return all (_subset(head[k],modifier[k]) for k in keys)



def combinations(head,dependent):
    """
    
    """
    l = []
    
    if dependent == False: #head = \bot
        l.append(("IGNORE",""))
        
    if head:
        for s in head.keys():
            if is_app_applicable(head,dependent,s):
                l.append(("APP_",s))
    if dependent:
        for s in dependent.keys():
            if is_mod_applicable(head,dependent,s):
                l.append(("MOD_",s))
    return l
    

class CombinationCache:
    
    def __init__(self):
        self.cache = dict()
    def combinations(self,head,head_str,dependent,dependent_str):
        try:
            return self.cache[(head_str,dependent_str)]
        except KeyError:
            combis = combinations(head,dependent)
            self.cache[(head_str,dependent_str)] = combis
            return combis
            

def number_of_open_sources(typ):
    if typ == False:
        return float("inf")
    if typ == None:
        return 0
    return len(typ.keys())



