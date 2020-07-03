# This file is based on is based on https://github.com/explosion/spaCy/blob/master/spacy/displacy/templates.py
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

# Setting explicit height and max-width: none on the SVG is required for
# Jupyter to render it properly in a cell

TPL_DEP_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="{lang}" id="{id}" class="displacy" width="{width}" height="{height}" direction="{dir}" style="max-width: none; height: {height}px; color: {color}; background: {bg}; font-family: {font}; direction: {dir}">{content}</svg>
"""


TPL_DEP_WORDS = """
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="{y}">
    <tspan class="displacy-word" fill="currentColor" x="{x}">{text}</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="{x}">{tag}</tspan>
</text>
"""


TPL_DEP_ARCS = """
<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-{id}-{i}" stroke-width="{stroke}px" d="{arc}" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-{id}-{i}" class="displacy-label" startOffset="50%" side="{label_side}" fill="currentColor" text-anchor="middle">{label}</textPath>
    </text>
    <path class="displacy-arrowhead" d="{head}" fill="currentColor"/>
</g>
"""

TPL_AM_DEP_ARCS = """
<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-{id}-{i}" stroke-width="{stroke}px" d="{arc}" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-{id}-{i}" class="displacy-label" startOffset="50%" side="{label_side}" fill="currentColor" text-anchor="middle">{operation}<tspan dy="2" font-size=".8em">{source}</tspan></textPath>
    </text>
    <path class="displacy-arrowhead" d="{head}" fill="currentColor"/>
</g>
"""


TPL_FIGURE = """
<figure style="margin-bottom: 6rem">{content}</figure>
"""

TPL_TITLE = """
<h2 style="margin: 0">{title}</h2>
"""



TPL_PAGE = """
<!DOCTYPE html>
<html lang="{lang}">
    <head>
        <title>displaCy</title>
    </head>
    <body style="font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: {dir}">{content}</body>
</html>
"""
