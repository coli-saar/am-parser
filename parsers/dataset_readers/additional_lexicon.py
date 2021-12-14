from typing import Tuple, Dict, List

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError


class Lexicon:
    """
    Mapping strings to unique ids and back.
    """

    UNK = "<UNK>"
    def __init__(self, file_name : str):
        self.s2i = dict()
        self.i2s = [Lexicon.UNK]
        with open(file_name) as fil:
            for i,line in enumerate(fil):
                line = line.rstrip("\n")
                self.i2s.append(line)
                self.s2i[line] = i+1

    def vocab_size(self) -> int:
        return len(self.i2s)

    def get_id(self, s : str):
        if s in self.s2i:
            return self.s2i[s]
        return 0

    def __iter__(self):
        return iter(self.s2i.items())

    def add(self, entries: List[str]) -> None:
        """
        Add new entries to end of vocabulary.
        :param entries:
        :return:
        """
        n = self.vocab_size()
        for i, e in enumerate(entries):
            self.i2s.append(e)
            self.s2i[e] = n+i



class AdditionalLexicon(Registrable):
    """
    A class with a lexicon for things like
     - edge labels
     - graph constants
     - (term) types
    this is needed because when the context is gathered in the DatasetReader, the mapping
    between these things and ids is not created yet.
    """
    POSSIBLE_KEYS = {"edge_labels", "term_types", "constants", "lex_labels"}

    def __init__(self, sublexica : Dict[str, str]):
        super().__init__()

        if not set(sublexica.keys()).issubset(AdditionalLexicon.POSSIBLE_KEYS):
            raise ConfigurationError(f"Unkown keys used: {sublexica.keys()}, I only know {AdditionalLexicon.POSSIBLE_KEYS}")

        self.sublexica = { name : Lexicon(path) for name, path  in sublexica.items()}

    def get_id(self, sublexicon : str, s : str) -> int:
        return self.sublexica[sublexicon].get_id(s)

    def get_str_repr(self, sublexicon : str, id : int) -> str:
        return self.sublexica[sublexicon].i2s[id]

    def contains(self, sublexicon:str, s : str) -> bool:
        return s in self.sublexica[sublexicon].s2i

    def vocab_size(self, sublexicon : str) -> int:
        return self.sublexica[sublexicon].vocab_size()