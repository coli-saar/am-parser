#
# Copyright (c) 2020 Saarland University.
#
# This file is part of AM Parser
# (see https://github.com/coli-saar/am-parser/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import subprocess
from typing import List, Dict, Tuple, Iterable, Union, Optional

from dataclasses import dataclass
import os
import multiprocessing as mp

from ...am_algebra.new_amtypes import AMType
from ...svg.dot_tools import penman_to_dot, parse_penman
from ...svg.render import DependencyRenderer


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

    def __eq__(self, other):
        if not isinstance(other, AMSentence):
            return False
        if len(self.words) != len(other.words):
            return False
        if self.attributes != other.attributes:
            return False

        return all( w==o for w,o in zip(self.words, other.words))

    def normalize_types(self) -> "AMSentence":
        """
        Parse the types and convert them to strings again to normalize them.
        :return:
        """
        return AMSentence([Entry(word.token, word.replacement, word.lemma, word.pos_tag, word.ner_tag, word.fragment, word.lexlabel,
                                 str(AMType.parse_str(word.typ)), word.head, word.label, word.aligned,word.range)
                           for i,word in enumerate(self.words)],self.attributes)

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

    def fix_dev_edge_labels(self) -> "AMSentence":
        """
        Return a copy of this sentence where edge labels that are "_" are replaced by "IGNORE".
        :return:
        """
        labels = self.get_edge_labels()
        return AMSentence([Entry(word.token, word.replacement, word.lemma, word.pos_tag, word.ner_tag, word.fragment, word.lexlabel,
                                 word.typ, word.head, labels[i], word.aligned,word.range)
                           for i,word in enumerate(self.words)],self.attributes)

    def set_lexlabels(self, labels : List[str]) -> "AMSentence":
        assert len(labels) == len(self.words), f"number of lexical labels must agree with number of words but got {len(labels)} and {len(self.words)}"
        return AMSentence([Entry(word.token, word.replacement, word.lemma, word.pos_tag, word.ner_tag, word.fragment, labels[i],
                                 word.typ, word.head, word.label, word.aligned,word.range)
                           for i,word in enumerate(self.words)],self.attributes)

    def set_labels(self, labels : List[str]) -> "AMSentence":
        assert len(labels) == len(self.words), f"number of lexical labels must agree with number of words but got {len(labels)} and {len(self.words)}"
        return AMSentence([Entry(word.token, word.replacement, word.lemma, word.pos_tag, word.ner_tag, word.fragment, word.lexlabel,
                                 word.typ, word.head, labels[i], word.aligned,word.range)
                           for i,word in enumerate(self.words)],self.attributes)

    def set_supertags(self, supertags : List[str]):
        assert len(supertags) == len(self.words), f"number of supertags must agree with number of words but got {len(supertags)} and {len(self.words)}"
        split = [ tag.split("--TYPE--") for tag in supertags]
        return AMSentence([Entry(word.token, word.replacement, word.lemma, word.pos_tag, word.ner_tag, split[i][0], word.lexlabel,
                                 split[i][1], word.head, word.label, word.aligned,word.range)
                           for i,word in enumerate(self.words)],self.attributes)

    def set_supertag_tuples(self, supertags : List[Tuple[str,str]]):
        assert len(supertags) == len(self.words), f"number of supertags must agree with number of words but got {len(supertags)} and {len(self.words)}"
        return AMSentence([Entry(word.token, word.replacement, word.lemma, word.pos_tag, word.ner_tag, supertags[i][0], word.lexlabel,
                                 supertags[i][1], word.head, word.label, word.aligned,word.range)
                           for i,word in enumerate(self.words)],self.attributes)

    def set_heads(self, heads : List[int]) -> "AMSentence":
        assert len(heads) == len(self.words), f"number of heads must agree with number of words but got {len(heads)} and {len(self.words)}"
        assert all( h >= 0 and h <= len(self.words) for h in heads), f"heads must be in range 0 to {len(self.words)} but got heads {heads}"

        return AMSentence([Entry(word.token, word.replacement, word.lemma, word.pos_tag, word.ner_tag, word.fragment, word.lexlabel,
                                 word.typ, heads[i], word.label, word.aligned, word.range)
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

    def strip_annotation(self) -> "AMSentence":
        return AMSentence([Entry(word.token, word.replacement, word.lemma, word.pos_tag, word.ner_tag, "_", "_",
                                 "_", 0, "IGNORE", word.aligned, word.range)
                           for word in self.words],self.attributes)

    def children_dict(self) -> Dict[int, List[int]]:
        """
        Return dictionary of children, 1-based.
        :return:
        """
        r = dict()
        for i in range(len(self.words)):
            head = self.words[i].head # head is 1-based
            if self.words[i].label != "IGNORE":
                if head not in r:
                    r[head] = []
                r[head].append(i+1) #make current position 1-based
        return r

    def get_root(self) -> Optional[int]:
        """
        Returns the index of the root, 0-based.
        :return:
        """
        for i,e in enumerate(self.words):
            if e.head == 0 and e.label == "ROOT":
                return i

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

    def to_dot(self):
        from parsers.svg.dot_tools import penman_to_dot
        r = []
        index_to_node_name = dict()
        for i,word in enumerate(self.words):
            if word.fragment == "_":
                continue

            cluster, a_node = penman_to_dot(word.fragment, word.lexlabel, word.lemma, word.token, word.replacement, word.pos_tag, "cl"+str(i)+"_")
            index_to_node_name[i] = a_node
            r.append("subgraph cluster"+str(i)+" { "+cluster+' label="'+word.token+'\\n'+word.typ+ '";}\n')

        for i, word in enumerate(self.words):
            head = word.head
            if head != 0:
                r.append(index_to_node_name[head-1] +" -> " + index_to_node_name[i] + " [ltail=cluster"+ str(head-1)+", lhead=cluster"+str(i)+', label="' + word.label+ '"];\n')

        return "digraph { compound=true; \n" +"\n".join(r) +"}"

    def to_tex_svg(self, directory, prefix=""):
        from parsers.svg.dot_tools import penman_to_dot, parse_penman, compile_dot
        def escape_chars(string: str) -> str:
            if string in "#$&%":
                string = "\\" + string
            # todo what if '\'
            # todo replace special char in longer string ('5$'->'5\$')
            # todo escape underscores? can this cause problems?
            return string.replace("_","\\_")
        r = """\\documentclass{standalone}
        \\usepackage[utf8]{inputenc}
        \\usepackage{tikz-dependency}
        \\usepackage{microtype}%svg weird output (second char lost in ligatures, idk why)
        \\DisableLigatures[f]{encoding = *, family = *} %% <- only disables f-ligatures
        \\usepackage{amsmath}
        \\usepackage{amsfonts}

        \\begin{document}

        \\newcommand{\\s}[1]{$_\\textsc{#1}$}

        \\begin{dependency}
        \\begin{deptext}[column sep=1.0cm]"""
        space = "\\&".join(" " for _ in self.words) + "\\\\ \n"

        r += "\\&".join(escape_chars(w.token) for w in self.words) + "\\\\ \n"
        r += space
        pretty_types = [ str(t.typ) for t in self.words]
        r += "\\&".join(t.replace("_","\\_") if t != "_" else "$\\bot$" for t in pretty_types) + "\\\\ \n"
        r += "\\&".join("\\hspace{1.2cm}" for _ in self.words) + "\\\\ \n"
        r += space
        r += "\end{deptext}\n"


        dot_filenames = []
        for i,word in enumerate(self.words,1):
            head = word.head
            if head == 0:
                if word.label == "ROOT":
                    r += "\\deproot{"+str(i)+"}{root}\n"
                else:
                    continue
            else:
                op, source = word.label.split("_")
                r += "\\depedge{"+str(head)+"}{"+str(i)+"}{"+op+"\s{"+source+"}}\n"

            graph_fragment = parse_penman(word.fragment)
            cluster, _ = penman_to_dot(graph_fragment, word.lexlabel, word.lemma, word.token, word.replacement, word.pos_tag, "n")
            fname = os.path.join(directory, "w"+str(i))
            with open(fname+".dot","w") as f:
                graphstyle = 'margin=0; bgcolor=transparent; node [fontsize=18,margin="0.05,0.005"]; '
                if len(graph_fragment.instances()) == 1:
                    #make smaller graph, otherwise node will look too large.
                    f.write('digraph{ graph [size="0.4,0.4"]; ' + graphstyle + cluster + "}")
                else:
                    f.write('digraph{ graph [size="0.8,0.8"]; ' + graphstyle + cluster + "}")
            dot_filenames.append(fname)

            #os.system("dot -Tpdf "+fname+".dot -o "+fname+".pdf")
            r += "\\node (n"+str(i)+") [below of = \wordref{5}{"+str(i)+"}] {\\includegraphics{w"+str(i)+".pdf}};\n"

        r += "\end{dependency} \end{document}"
        fname = os.path.join(directory, prefix+"sentence")

        #compile dot graphs in parallel.
        with mp.Pool(8) as p:
            p.map(compile_dot, dot_filenames)

        with open(fname+".tex","w") as g:
            g.write(r)

        subprocess.run("pdflatex -interaction=batchmode "+ fname+".tex" + " > /dev/null 2>&1", shell=True, cwd=directory)
        #os.system("./pdf2svg " + fname + ".pdf " + fname+".svg")
        os.system("inkscape -l " + fname + ".svg " + fname+".pdf")
        os.system("cat "+fname+".svg | tr -s ' ' > "+fname+"2.svg") #svg contains a lot of spaces, strip them away.
        with open(fname+"2.svg") as f:
            return f.read()

    def displacy_svg(self):
        renderer = DependencyRenderer({"compact" : True})
        root_node = 0
        d = {"words" :  [ {"text" : w.token, "tag" : str(AMType.parse_str(w.typ)) if w.typ != "_" else "⊥"} for w in self.words] }
        d["arcs"] = []
        for i, word in enumerate(self.words):
            if word.head != 0:
                if i < word.head-1:
                    start = i
                    end = word.head-1
                else:
                    start = word.head-1
                    end = i
                d["arcs"].append({"start": start, "end" : end, "label" : word.label, "dir" : "left" if i < word.head-1 else "right"})

            if word.head == 0 and word.label == "ROOT":
                root_node = i

            if word.fragment == "_":
                d["words"][i]["supertag"] = ""
                continue
            graph_fragment = parse_penman(word.fragment)
            cluster, _ = penman_to_dot(graph_fragment, word.lexlabel, word.lemma, word.token, word.replacement, word.pos_tag, "n")

            if len(graph_fragment.instances()) == 1:
                #make smaller graph, otherwise node will look too large.
                d["words"][i]["supertag"] = 'digraph{ graph [size="0.7,0.7"]; margin=0; bgcolor=transparent; ' + cluster + "}"
            else:
                d["words"][i]["supertag"] =  'digraph{ graph [size="1.4,1.4"]; margin=0; bgcolor=transparent; ' + cluster + "}"

        d["root"] = root_node
        return renderer.render([d])

def from_raw_text(rawstr: str, words: List[str], add_art_root: bool, attributes: Dict, contract_ne: bool) -> AMSentence:
    """
    Create an AMSentence from raw text, without token ranges and stuff
    :param words:
    :param add_art_root:
    :param attributes:
    :param contract_ne: shall we contract named entites, e.g. Barack Obama --> Barack_Obama. Should be done only for AMR.
    :return:
    """
    entries = []
    # use spacy lemmas and tags
    from parsers.components.spacy_interface import run_spacy, lemma_postprocess, ne_postprocess, is_number

    spacy_doc = run_spacy([words])
    ne = []
    for i in range(len(words)):
        word = words[i]
        lemma = lemma_postprocess(word, spacy_doc[i].lemma_)
        if contract_ne:
            if spacy_doc[i].ent_type_ not in ["QUANTITY", "PERCENT", "CARDINAL", "MONEY"]:
                if spacy_doc[i].ent_iob_ == "B":
                    if len(ne) > 0:
                        ent_typ = ne_postprocess(spacy_doc[i - 1].ent_type_)
                        e = Entry("_".join(ne), is_number(ent_typ),
                                  lemma_postprocess(words[i - 1], spacy_doc[i - 1].lemma_), spacy_doc[i - 1].tag_, "O",
                                  "_", "_", "_", 0, "IGNORE", True, None)
                        entries.append(e)
                    ne = [word]
                elif spacy_doc[i].ent_iob_ == "I":
                    ne.append(word)

            if len(ne) > 0:
                if (i == len(words) - 1 or i + 1 < len(words) and spacy_doc[i + 1].ent_iob_ != "I"):
                    ent_typ = ne_postprocess(spacy_doc[i].ent_type_)

                    e = Entry("_".join(ne), is_number(ent_typ), lemma, spacy_doc[i].tag_, "O", "_", "_", "_", 0,
                              "IGNORE", True, None)
                    entries.append(e)
                    ne = []
            else:
                #ne_postprocess(spacy_doc[i].ent_type_)
                replacement = "_" if word == word.lower() else word.lower()
                e = Entry(word, replacement, lemma, spacy_doc[i].tag_,"O" , "_", "_", "_", 0, "IGNORE", True,
                          None)
                entries.append(e)

        else:  # don't contract NEs
            # ne_postprocess(spacy_doc[i].ent_type_)
            replacement = "_" if word == word.lower() else word.lower()
            e = Entry(word, replacement, lemma, spacy_doc[i].tag_, "O", "_", "_", "_", 0, "IGNORE", True,
                      None)
            entries.append(e)

    if add_art_root:
        entries.append(
            Entry("ART-ROOT", "_", "ART-ROOT", "ART-ROOT", "ART-ROOT", "_", "_", "_", 0, "IGNORE", True, None))
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


