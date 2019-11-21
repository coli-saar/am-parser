#Takes a graphbank and procudes for every graph a pdf file in the current directory
#Before running: set path to mtool!
mtool="python3 /home/matthias/bin/mtool_code/main.py"

#First argument: mtool file type either amr, dm, psd, pas, mrp (and eds, ud and ucca -- I didn't test those though, ml)
#Second argument: prefix used for output files. For instance, chosing "AMR" results in the AMR-01.pdf, AMR-02.pdf etc.
#Third argument: file with graphs in specific format (see first argument)

format="$1"
prefix="$2"

$mtool --read $format --write dot --normalize all "$3" "/tmp/$prefix.dot"

dot -Tpdf "/tmp/$prefix.dot" | csplit --quiet --elide-empty-files --prefix="$prefix-" - "/%%EOF/+1" "{*}" -b "%02d.pdf"

rm "/tmp/$prefix.dot"
