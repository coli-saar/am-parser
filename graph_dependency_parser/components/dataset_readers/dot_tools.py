def relex(label, lex_label, lemma, form, replacement, pos):
    lex_label = lex_label.replace("$LEMMA$", lemma)
    lex_label = lex_label.replace("$FORM$", form)
    lex_label = lex_label.replace("$REPL$", replacement)
    lex_label = lex_label.replace("$POS$", pos)
    return label.replace("--LEX--", lex_label)


def penman_to_dot(graph_fragment, lex_label, lemma, form, replacement, pos, prefix="n"):
    """
    Converts a supertag to a little dot graph.
    """

    import penman
    g = penman.decode(graph_fragment)
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

