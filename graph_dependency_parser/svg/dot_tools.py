def relex(label, lex_label, lemma, form, replacement, pos):
    lex_label = lex_label.replace("$LEMMA$", lemma)
    lex_label = lex_label.replace("$FORM$", form)
    lex_label = lex_label.replace("$REPL$", replacement)
    lex_label = lex_label.replace("$POS$", pos)
    return label.replace("--LEX--", lex_label)

def parse_penman(graph_fragment):
    import penman
    return penman.decode(graph_fragment)

def penman_to_dot(graph_fragment, lex_label, lemma, form, replacement, pos, prefix="n"):
    """
    Converts a supertag to a little dot graph.
    """

    import penman
    if isinstance(graph_fragment, str):
        g = penman.decode(graph_fragment)
    else:
        g = graph_fragment
    name2name = dict()
    accounted_for = set()
    counter = 0
    r = ""

    for f,rel, to in g.triples:

        if f not in name2name:
            new_name = prefix+str(counter)
            counter += 1
            name2name[f] = new_name

        if rel != ":instance" and to not in name2name:
            new_name = prefix+str(counter)
            counter += 1
            name2name[to] = new_name

    for f,rel, to in g.triples:

        if rel == ":instance":
            is_root = f == g.top
            if to is None:
                source = f.split("<")[1][:-1]
                if is_root:
                    r += name2name[f] + ' [label="' + source + '", fontcolor="red", style="bold"];\n'
                else:
                    r += name2name[f] + ' [label="' + source + '", fontcolor="red"];\n'
            else:
                label = relex(to, lex_label, lemma, form, replacement, pos)
                if is_root:
                    r += name2name[f] + ' [style="bold", label="' + label + '"];\n'
                else:
                    r += name2name[f] + ' [label="' + label + '"];\n'
            accounted_for.add(name2name[f])
        else:

            r += name2name[f] + " -> " + name2name[to] + ' [label="' + rel[1:] + '"];\n'

    assert set(accounted_for) == set(name2name.values())

    return r, name2name[g.top]


import os
import subprocess
import re

def compile_dot(fname):
    os.system("dot -Tpdf "+fname+".dot -o "+fname+".pdf")

def get_dot(graph, format):
    with subprocess.Popen("dot -T"+format, shell=True, stdout=subprocess.PIPE,stdin=subprocess.PIPE) as proc:
        proc.stdin.write(bytes(graph,"utf8"))
        proc.stdin.close()
        return bytes.decode(proc.stdout.read())  # output of shell commmand as string

def dot_strip_svg_header(svg):
    return "\n".join(svg.split("\n")[3:])


class DotSVG:
    """
    Quick, dirty and limited method to manipulate the output of dot -Tsvg
    """

    width_pattern = re.compile('width="([^"]+)"')
    height_pattern = re.compile('height="([^"]+)"')

    def __init__(self, dot_script):
        self.s = get_dot(dot_script, "svg")

    def get_width(self):
        m = re.search(self.width_pattern, self.s)
        return m.group(1) #e.g. 89pt

    def get_height(self):
        m = re.search(self.height_pattern, self.s)
        return m.group(1)

    def get_str(self):
        return self.s

    def get_str_without_header(self):
        return dot_strip_svg_header(self.s)

    def set_xy(self, x,y):
        self.s = self.s.replace("<svg",'<svg x="'+x+'" y="'+y+'"')

    def set_width(self, w):
        self.s = re.sub(self.width_pattern, 'width="'+w+'"', self.s)

    def set_height(self, h):
        self.s = re.sub(self.height_pattern, 'height="'+h+'"', self.s)



