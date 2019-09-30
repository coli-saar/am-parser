from typing import List, Dict, Tuple, Iterable, Union

from dataclasses import dataclass

from graph_dependency_parser.components.spacy_interface import run_spacy


@dataclass(frozen=True)
class Entry:
    token: str
    replacement: str
    lemma: str
    pos_tag: str
    ner_tag: str
    fragment: str
    lexlabel: str
    typ: str
    head: int
    label: str
    aligned: bool
    range: Union[str,None]

    def __iter__(self):
        return iter([self.token, self.replacement, self.lemma, self.pos_tag, self.ner_tag, self.fragment, self.lexlabel,
                     self.typ, self.head, self.label, self.aligned, self.range])


@dataclass
class AMSentence:
    """Represents a sentence"""
    words: List[Entry]
    attributes: Dict[str, str]

    def __iter__(self):
        return iter(self.words)

    def __index__(self, i):
        """Zero-based indexing."""
        return self.words[i]

    def get_tokens(self, shadow_art_root) -> List[str]:
        r = [word.token for word in self.words]
        if shadow_art_root and r[-1] == "ART-ROOT":
            r[-1] = "."
        return r

    def get_replacements(self) -> List[str]:
        return [word.replacement for word in self.words]

    def get_pos(self) -> List[str]:
        return [word.pos_tag for word in self.words]

    def get_lemmas(self) -> List[str]:
        return [word.lemma for word in self.words]

    def get_ner(self) -> List[str]:
        return [word.ner_tag for word in self.words]

    def get_supertags(self) -> List[str]:
        return [word.fragment+"--TYPE--"+word.typ for word in self.words]

    def get_lexlabels(self) -> List[str]:
        return [word.lexlabel for word in self.words]

    def get_ranges(self) -> List[str]:
        return [word.range for word in self.words]

    def get_heads(self)-> List[int]:
        return [word.head for word in self.words]

    def get_edge_labels(self) -> List[str]:
        return [word.label if word.label != "_" else "IGNORE" for word in self.words] #this is a hack :(, which we need because the dev data contains _

    def set_lexlabels(self, labels : List[str]) -> "AMSentence":
        assert len(labels) == len(self.words), f"number of lexical labels must agree with number of words but got {len(labels)} and {len(self.words)}"
        return AMSentence([Entry(word.token, word.replacement, word.lemma, word.pos_tag, word.ner_tag, word.fragment, labels[i],
                                word.typ, word.head, word.label, word.aligned)
                           for i,word in enumerate(self.words)],self.attributes)


    @staticmethod
    def get_bottom_supertag() -> str:
        return "_--TYPE--_"

    @staticmethod
    def split_supertag(supertag : str) -> Tuple[str,str]:
        return tuple(supertag.split("--TYPE--",maxsplit=1))

    def attributes_to_list(self) -> List[str]:
        return [ f"#{key}:{val}" for key,val in self.attributes.items()]

    def check_validity(self):
        """Checks if representation makes sense, doesn't do AM algebra type checking"""
        assert len(self.words) > 0, "Sentence is empty"
        for entry in self.words:
            assert entry.head in range(len(self.words) + 1), f"head of {entry} is not in sentence range"
        has_root = any(w.label == "ROOT" and w.head == 0 for w in self.words)
        if not has_root:
            assert all((w.label == "IGNORE" or w.label=="_") and w.head == 0 for w in self.words), f"Sentence doesn't have a root but seems annotated with trees:\n {self}"

    def __str__(self):
        r = []
        if self.attributes:
            r.append("\n".join(f"#{attr}:{val}" for attr, val in self.attributes.items()))
        for i, w in enumerate(self.words, 1):
            fields = list(w)
            if fields[-1] is None:
                fields = fields[:-1] #when token range not present -> remove it
            r.append("\t".join([str(x) for x in [i] + fields]))
        return "\n".join(r)

    def is_annotated(self):
        return not all((w.label == "_" or w.label == "IGNORE") and w.head == 0 for w in self.words)

    def __len__(self):
        return len(self.words)


def from_raw_text(rawstr : str, words: List[str], add_art_root : bool, attributes: Dict) -> AMSentence:
    """
    Create an AMSentence from raw text, without token ranges and stuff
    :param words:
    :param add_art_root:
    :param attributes:
    :return:
    """
    entries = []
    #use spacy lemmas and tags
    spacy_doc = run_spacy([words])
    for i, word in zip(range(len(words)), words):
        e = Entry(word,"_",spacy_doc[i].lemma_,spacy_doc[i].tag_,"O","_","_","_",0,"IGNORE",True,None)
        entries.append(e)
    if add_art_root:
        entries.append(Entry("ART-ROOT","_","ART-ROOT","ART-ROOT","ART-ROOT","_","_","_",0,"IGNORE",True,None))
    attributes["raw"] = rawstr
    sentence = AMSentence(entries, attributes)
    sentence.check_validity()
    return sentence


def parse_amconll(fil, validate:bool = True) -> Iterable[AMSentence]:
    """
    Reads a file and returns a generator over AM sentences.
    :param fil:
    :return:
    """
    expect_header = True
    new_sentence = True
    entries = []
    attributes = dict()
    for line in fil:
        line = line.rstrip("\n")
        if line.strip() == "":
            # sentence finished
            if len(entries) > 0:
                sent = AMSentence(entries, attributes)
                if validate:
                    sent.check_validity()
                yield sent
            new_sentence = True

        if new_sentence:
            expect_header = True
            attributes = dict()
            entries = []
            new_sentence = False
            if line.strip() == "":
                continue

        if expect_header:
            if line.startswith("#"):
                key, val = line[1:].split(":", maxsplit=1)
                attributes[key] = val
            else:
                expect_header = False

        if not expect_header:
            fields = line.split("\t")
            assert len(fields) == 12 or len(fields) == 13
            if len(fields) == 12 : #id + entry but no token ranges
                entries.append(
                    Entry(fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], fields[7], fields[8],
                          int(fields[9]), fields[10], bool(fields[11]),None))
            elif len(fields) == 13:
                entries.append(
                    Entry(fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], fields[7], fields[8],
                          int(fields[9]), fields[10], bool(fields[11]),fields[12]))


