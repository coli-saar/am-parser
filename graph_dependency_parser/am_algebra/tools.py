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

# NOTE: these functions are not used in "the" AM parser but only in the transition-based AM parser.

from typing import Optional, Tuple, List, Dict, Set, Iterable

from graph_dependency_parser.components.dataset_readers.amconll_tools import AMSentence, Entry
from .new_amtypes import AMType, ReadCache, NonAMTypeException
from .tree import Tree


def is_welltyped(sent: AMSentence) -> bool:
    return get_tree_type(sent) is not None


def get_tree_type(sent : AMSentence) -> Optional[AMType]:
    """
    Get the term type at the root of sent, or None if not well-typed.
    :param sent:
    :return:
    """
    root = sent.get_root()
    if root is None:
        return None

    term_types = get_term_types(sent)
    return term_types[root]


def get_term_types(sent: AMSentence) -> List[Optional[AMType]]:
    """
    Return a list of length len(sent), where each element is the term type
    of the subtree rooted in the respective token, or None if the subtree is not well-typed.
    :param sent:
    :return:
    """
    deptree = Tree.from_am_sentence(sent)
    cache = ReadCache()
    term_types = [None for _ in sent.words]

    def determine_tree_type(node: Tuple[int, Entry], children: List[Tuple[Optional[AMType],str]]) -> Tuple[Optional[AMType],str]:
        try:
            lextyp = cache.parse_str(node[1].typ)
        except NonAMTypeException:
            return None, node[1].label

        apply_children : Dict[str, AMType] = dict() # maps sources to types of children

        for child_typ, label in children:
            if child_typ is None: # one child is not well-typed
                return None, node[1].label

            if "_" in label:
                source = label.split("_")[1]
                if label.startswith("MOD") and not lextyp.can_be_modified_by(child_typ, source):
                    return None, node[1].label
                elif label.startswith("APP"):
                    apply_children[source] = child_typ
            else:
                if label == "IGNORE":
                    if not child_typ.is_bot:
                        return None, node[1].label

                elif label == "ROOT":
                    if node[1].head == -1:
                        return child_typ, node[1].label
                    else:
                        return None, node[1].label
                else:
                    raise ValueError("Nonsensical edge label: "+label)

        typ = lextyp
        changed = True
        while changed:
            changed = False
            remove = []
            for o in apply_children:
                if typ.can_apply_to(apply_children[o], o):
                    typ = typ.perform_apply(o)
                    remove.append(o)
                    changed = True

            for source in remove:
                del apply_children[source]

        if apply_children == dict():
            term_types[node[0]-1] = typ
            return typ, node[1].label

        return None, node[1].label

    deptree.fold(determine_tree_type)
    return term_types



