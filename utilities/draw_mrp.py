import os
import sys
import subprocess
import tempfile

data = sys.stdin.readlines()
data = "\n".join(data)


tmpfile = tempfile.mktemp()
tmpdot = tempfile.mktemp("dot")
tmppdf = tempfile.mktemp("pdf")

with open(tmpfile, "w") as f:
    f.write(data)

os.system(f"python ../mtool/main.py --n 1 --ids --strings --read mrp --write dot {tmpfile} {tmpdot}")
os.system(f"dot -Tpdf {tmpdot} > {tmppdf}")
os.system(f"open {tmppdf}")