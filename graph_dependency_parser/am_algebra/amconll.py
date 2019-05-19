from collections import Counter, OrderedDict, UserList

class ConllEntry:
    def __init__(self, id, form, replacement, lemma, pos, ne, delex_supertag, lex_label, typ, parent_id, edge_label, aligned=True):
        self.aligned = aligned
        self.id = id
        self.form = form
        self.replacement = replacement
        self.lemma = lemma
        self.pos = pos
        self.ne = ne
        self.delex_supertag = delex_supertag
        self.lex_label = lex_label
        self.typ = typ
        self.parent_id = parent_id
        self.edge_label = edge_label

        self.supertags = [] #will contain triples score, delex, type that the fixed tree decoder uses

        self.pred_parent_id = None
        self.pred_edge_label = None

        
    def getTag(self):
        return (self.delex_supertag, self.typ)
    
    
    def copy(self):
        e = ConllEntry(self.id, self.form, self.replacement, self.lemma, self.pos, self.ne, self.delex_supertag, self.lex_label, self.typ, self.parent_id, self.edge_label, self.aligned)
        e.supertags = self.supertags
        return e

    def __str__(self):
        values = [str(self.id), self.form, self.replacement, self.lemma, self.pos, self.ne, self.delex_supertag, self.lex_label, str(self.typ), str(self.pred_parent_id), self.pred_edge_label, str(self.aligned)]
        return '\t'.join(['_' if v is None else v for v in values])


class ConllSent(UserList):
    """
    A class for representing sentences. Each sentence consists of several ConllEntries (one per word + a representation of the artificial root at position 0).
    A sentence belongs to a task and may have other sentence-wide valid attributes (e.g. untokenized string).
    """
    def __init__(self, heads, label_scores, root):
        super().__init__()
        self.attrs = []
        self.root = root #root index
        self.heads = heads #head for each word. The first entry should be -1 (because the artificial root has no root)
        self.label_scores = label_scores #access from -> to -> label. That is, the dimensions are Tokens x Tokens x Label Types
    def add_attr(self, attr):
        self.attrs.append(attr)
    def get_attrs(self):
        return self.attrs
    def copy(self):
        n = ConllSent()
        n.attrs = list(self.attrs)
        for e in self:
            n.append(e.copy())
        return n

def write_conll(fn, conll_gen):
    """
    Takes a file object and an iterable of
    :param fn:
    :param conll_gen:
    :return:
    """
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for attr in sentence.get_attrs():
                fh.write(attr)
                if not (attr.endswith("\n")):
                    fh.write("\n")
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')