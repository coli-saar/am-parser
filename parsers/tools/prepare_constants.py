import argparse
import os

import sys
from typing import Dict, Set

sys.path.append(".")
from parsers.dataset_readers.amconll_tools import parse_amconll, AMSentence
from parsers.am_algebra import ReadCache, NonAMTypeException, AMType
from parsers.am_algebra.tree import Tree
from parsers.am_algebra.tools import get_term_types, is_welltyped

optparser = argparse.ArgumentParser(add_help=True,
                                    description="reads amconll file and produces two files with the possible graph constants and with the possible types. Ivents new constants where needed.")
optparser.add_argument("output", type=str, help="Output directory")

optparser.add_argument('--corpora',
                       nargs="+",
                       default=[],
                       help='Corpora to read, typically the train.amconll and gold-dev.amconll file')



def invent_supertag(type : AMType) -> str:
    fragment = "(r<root> / --LEX--"
    for i,origin in enumerate(type.origins):
        fragment += f" :invalid-label (n{i}<{origin}>)"
    fragment += ")--TYPE--" + str(type)
    return fragment

args = optparser.parse_args()

supertags : Dict[AMType, Set[str]] = dict() #maps lexical types to supertags with --TYPE--
lexical_types : Set[AMType] = set()
term_types : Set[AMType] = set()
read_cache = ReadCache()
all_types : Set[AMType] = set()

mod_sources : Set[str] = set()

for corpus in args.corpora:
    with open(corpus) as f:
        trees = parse_amconll(f)
        for am_sentence in trees:
            if not is_welltyped(am_sentence):
                print("Skipping non-well-typed AMDep tree.")
                continue

            term_types.update(get_term_types(am_sentence))
            for entry in am_sentence.words:
                typ = read_cache.parse_str(entry.typ)
                lexical_types.add(typ)
                if typ not in supertags:
                    supertags[typ] = set()
                supertags[typ].add(entry.fragment + "--TYPE--" + str(typ))

                if entry.label.startswith("MOD_"):
                    mod_sources.add(entry.label.split("_")[1])

all_types = lexical_types | term_types
invented = 0
# Make up constants for types that we did not observe as lexical types
print("Inventing constants for term types...")
for unknown_type in term_types - lexical_types:
    print("Invented",invent_supertag(unknown_type))
    supertags[unknown_type] = {invent_supertag(unknown_type)}
    invented += 1

# Closure under requests:

changed = True
print("Checking for closure under requests...")
while changed:
    changed = False
    all_requests = set()
    for typ in all_types:
        for node in typ.nodes():
            all_requests.add(typ.get_request(node))

    for unknown_type in all_requests - all_types:
        print("Invented", invent_supertag(unknown_type))
        supertags[unknown_type] = {invent_supertag(unknown_type)}
        invented += 1
        changed = True
    all_types.update(all_requests)

# Mod sources
print("Identified the following mod sources", mod_sources)
simple_mod_types = {AMType.parse_str(f"({source})") for source in mod_sources}
for unknown_type in simple_mod_types - all_types:
    print("Invented", invent_supertag(unknown_type))
    supertags[unknown_type] = {invent_supertag(unknown_type)}
    invented += 1

all_types.update(simple_mod_types)

print("Invented", invented, "constants in total.")

# Write constants and types to files:
with open(os.path.join(args.output, "constants.txt"),"w") as f:
    for lex_type in sorted(supertags.keys(), key=lambda typ: str(typ)):
        for constant in sorted(supertags[lex_type]):
            f.write(constant)
            f.write("\n")

all_types_sorted = sorted(all_types, key=lambda t: str(t))
with open(os.path.join(args.output, "types.txt"),"w") as f:
    for t in all_types_sorted:
        f.write(str(t))
        f.write("\n")
