##
## Copyright (c) 2020 Saarland University.
##
## This file is part of AM Parser
## (see https://github.com/coli-saar/am-parser/).
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
mtool="python3 /home/matthias/bin/mtool_code/main.py"

#First argument: mtool file type either amr, dm, psd, pas, mrp (and eds, ud and ucca -- I didn't test those though, ml)
#Second argument: prefix used for output files. For instance, chosing "AMR" results in the AMR-01.pdf, AMR-02.pdf etc.
#Third argument: file with graphs in specific format (see first argument)

format="$1"
prefix="$2"

$mtool --read $format --write dot --normalize all "$3" "/tmp/$prefix.dot"

dot -Tpdf "/tmp/$prefix.dot" | csplit --quiet --elide-empty-files --prefix="$prefix-" - "/%%EOF/+1" "{*}" -b "%02d.pdf"

rm "/tmp/$prefix.dot"
