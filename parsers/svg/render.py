# This file is based on is based on https://github.com/explosion/spaCy/blob/master/spacy/displacy/render.py
# which is licensed under the MIT license (see LICENSE.txt in this folder)

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

# coding: utf8
from __future__ import unicode_literals


import uuid

from .templates import TPL_DEP_SVG, TPL_DEP_WORDS, TPL_AM_DEP_ARCS, TPL_FIGURE, TPL_PAGE

import multiprocessing as mp


DEFAULT_LANG = "en"
DEFAULT_DIR = "ltr"

def minify_html(html):
    """Perform a template-specific, rudimentary HTML minification for displaCy.
    Disclaimer: NOT a general-purpose solution, only removes indentation and
    newlines.
    html (unicode): Markup to minify.
    RETURNS (unicode): "Minified" HTML.
    """
    return html.strip().replace("    ", "").replace("\n", "")


def escape_html(text):
    """Replace <, >, &, " with their HTML encoded representation. Intended to
    prevent HTML errors in rendered displaCy markup.
    text (unicode): The original text.
    RETURNS (unicode): Equivalent text to be safely used within HTML.
    """
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    return text



class DependencyRenderer(object):
    """Render dependency parses as SVGs."""

    style = "dep"

    def __init__(self, options={}):
        """Initialise dependency renderer.
        options (dict): Visualiser-specific options (compact, word_spacing,
            arrow_spacing, arrow_width, arrow_stroke, distance, offset_x,
            color, bg, font)
        """
        self.compact = options.get("compact", False)
        self.word_spacing = options.get("word_spacing", 45)
        self.arrow_spacing = options.get("arrow_spacing", 12 if self.compact else 20)
        self.arrow_width = options.get("arrow_width", 6 if self.compact else 10)
        self.arrow_stroke = options.get("arrow_stroke", 2)
        self.distance = options.get("distance", 150 if self.compact else 175)
        self.offset_x = options.get("offset_x", 70)
        self.color = options.get("color", "#000000")
        self.bg = options.get("bg", "#ffffff")
        self.font = options.get("font", "Arial")
        self.direction = DEFAULT_DIR
        self.lang = DEFAULT_LANG

    def render(self, parsed, page=False, minify=False):
        """Render complete markup.
        parsed (list): Dependency parses to render.
        page (bool): Render parses wrapped as full HTML page.
        minify (bool): Minify HTML markup.
        RETURNS (unicode): Rendered SVG or HTML markup.
        """
        # Create a random ID prefix to make sure parses don't receive the
        # same ID, even if they're identical
        id_prefix = uuid.uuid4().hex
        rendered = []
        for i, p in enumerate(parsed):
            if i == 0:
                settings = p.get("settings", {})
                self.direction = settings.get("direction", DEFAULT_DIR)
                self.lang = settings.get("lang", DEFAULT_LANG)
            render_id = "{}-{}".format(id_prefix, i)
            svg = self.render_svg(render_id, p["words"], p["arcs"], p["root"])
            rendered.append(svg)
        if page:
            content = "".join([TPL_FIGURE.format(content=svg) for svg in rendered])
            markup = TPL_PAGE.format(
                content=content, lang=self.lang, dir=self.direction
            )
        else:
            markup = "".join(rendered)
        if minify:
            return minify_html(markup)
        return markup

    def render_svg(self, render_id, words, arcs, root):
        """Render SVG.
        render_id (int): Unique ID, typically index of document.
        words (list): Individual words and their tags.
        arcs (list): Individual arcs and their start, end, direction and label.
        root (int): root index
        RETURNS (unicode): Rendered SVG markup.
        """
        self.levels = self.get_levels(arcs)
        self.highest_level = len(self.levels)
        self.offset_y = self.distance / 6 * (self.highest_level+2) + self.arrow_stroke
        self.width = self.offset_x + len(words) * self.distance
        self.height = self.offset_y + 3 * self.word_spacing + 110

        self.id = render_id
        words_list = [self.render_word(w["text"], w["tag"], i) for i, w in enumerate(words)]
        supertags = [(w["supertag"],i) for i,w in enumerate(words)]

        with mp.Pool(6) as p:
            supertags = p.map(self.render_supertag, supertags)

        arcs = [
            self.render_arrow(a["label"], a["start"], a["end"], a["dir"], i)
            for i, a in enumerate(arcs)
        ]
        content = "".join(words_list) + "".join(supertags) + "".join(arcs) + self.render_root(root)
        return TPL_DEP_SVG.format(
            id=self.id,
            width=self.width,
            height=self.height,
            color=self.color,
            bg=self.bg,
            font=self.font,
            content=content,
            dir=self.direction,
            lang=self.lang,
        )

    def render_supertag(self, pair):
        """
        Renders the graph fragment for word i.
        """
        supertag_as_dot,i = pair
        from .dot_tools import DotSVG
        if supertag_as_dot == "":
            return ""

        dot_svg = DotSVG(supertag_as_dot)

        width = int(dot_svg.get_width()[:-2])

        dot_svg.set_xy(str(self.offset_x + i * self.distance - width/2 - 12), str(self.offset_y + self.word_spacing + 60))
        return dot_svg.get_str_without_header()

    def render_root(self, i):
        root_node = """<text class="displacy-root" fill="currentColor" text-anchor="middle" x="{x}" y="{y}">
            root
        </text>\n"""
        root_arrow = """<g class="displacy-arrow">
            <path class="displacy-arc" id="arrow-root-{id}" stroke-width="2px" d="M{start} L{end}" fill="none" stroke="currentColor"/>
        </g>\n"""

        if self.compact:
            y_pos = self.offset_y - (self.highest_level+1) * self.distance / 6
        else:
            y_pos = self.offset_y - (self.highest_level+1) * self.distance / 2
        x_pos = self.offset_x + i * self.distance
        return root_node.format(x=x_pos, y = y_pos) + root_arrow.format(id=self.id, start=",".join((str(x_pos), str(y_pos+5))), end = ",".join((str(x_pos), str(self.offset_y))))


    def render_word(self, text, tag, i):
        """Render individual word.
        text (unicode): Word text.
        tag (unicode): Part-of-speech tag.
        i (int): Unique ID, typically word index.
        RETURNS (unicode): Rendered SVG markup.
        """
        y = self.offset_y + self.word_spacing
        x = self.offset_x + i * self.distance
        if self.direction == "rtl":
            x = self.width - x
        html_text = escape_html(text)
        return TPL_DEP_WORDS.format(text=html_text, tag=tag, x=x, y=y)

    def render_arrow(self, label, start, end, direction, i):
        """Render individual arrow.
        label (unicode): Dependency label.
        start (int): Index of start word.
        end (int): Index of end word.
        direction (unicode): Arrow direction, 'left' or 'right'.
        i (int): Unique ID, typically arrow index.
        RETURNS (unicode): Rendered SVG markup.
        """
        if start < 0 or end < 0:
            error_args = dict(start=start, end=end, label=label, dir=direction)
            raise ValueError(**error_args)
        level = self.levels.index(end - start) + 1
        x_start = self.offset_x + start * self.distance + self.arrow_spacing
        if self.direction == "rtl":
            x_start = self.width - x_start
        y = self.offset_y
        x_end = (
                self.offset_x
                + (end - start) * self.distance
                + start * self.distance
                - self.arrow_spacing * (self.highest_level - level) / 4
        )
        if self.direction == "rtl":
            x_end = self.width - x_end
        y_curve = self.offset_y - level * self.distance / 2
        if self.compact:
            y_curve = self.offset_y - level * self.distance / 6
        if y_curve == 0 and len(self.levels) > 5:
            y_curve = -self.distance
        arrowhead = self.get_arrowhead(direction, x_start, y, x_end)
        arc = self.get_arc(x_start, y, y_curve, x_end)
        label_side = "right" if self.direction == "rtl" else "left"

        if "_" in label:
            op, source = label.split("_")
        else:
            op = label
            source = ""

        return TPL_AM_DEP_ARCS.format(
            id=self.id,
            i=i,
            stroke=self.arrow_stroke,
            head=arrowhead,
            operation=op,
            source=source,
            label_side=label_side,
            arc=arc,
        )

    def get_arc(self, x_start, y, y_curve, x_end):
        """Render individual arc.
        x_start (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        y_curve (int): Y-corrdinate of Cubic Bézier y_curve point.
        x_end (int): X-coordinate of arrow end point.
        RETURNS (unicode): Definition of the arc path ('d' attribute).
        """
        template = "M{x},{y} C{x},{c} {e},{c} {e},{y}"
        if self.compact:
            template = "M{x},{y} {x},{c} {e},{c} {e},{y}"
        return template.format(x=x_start, y=y, c=y_curve, e=x_end)

    def get_arrowhead(self, direction, x, y, end):
        """Render individual arrow head.
        direction (unicode): Arrow direction, 'left' or 'right'.
        x (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        end (int): X-coordinate of arrow end point.
        RETURNS (unicode): Definition of the arrow head path ('d' attribute).
        """
        if direction == "left":
            pos1, pos2, pos3 = (x, x - self.arrow_width + 2, x + self.arrow_width - 2)
        else:
            pos1, pos2, pos3 = (
                end,
                end + self.arrow_width - 2,
                end - self.arrow_width + 2,
            )
        arrowhead = (
            pos1,
            y + 2,
            pos2,
            y - self.arrow_width,
            pos3,
            y - self.arrow_width,
        )
        return "M{},{} L{},{} {},{}".format(*arrowhead)

    def get_levels(self, arcs):
        """Calculate available arc height "levels".
        Used to calculate arrow heights dynamically and without wasting space.
        args (list): Individual arcs and their start, end, direction and label.
        RETURNS (list): Arc levels sorted from lowest to highest.
        """
        levels = set(map(lambda arc: arc["end"] - arc["start"], arcs))
        return sorted(list(levels))
