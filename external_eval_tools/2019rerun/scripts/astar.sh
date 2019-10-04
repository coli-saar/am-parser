#!/bin/bash

usage="Runs the parser, relabeler, and smatch calculator. \n\n

Required argument: \n
\n\t     -d  subdirectory where the results should be put. 
\n\t     -a  path to alto jar file containing de.up.ling.irtg.experimental.astar.Astar 

\noptions:

\n\t	-b  use bias to speed up A* search (use negative bias; default 0.0)
\n\t	-t  number of threads to use in parser
\n\t    -h  bring up this menu
"

bias="0.0"
threads="1"

while getopts "d:a:b:t:h" opt; do
    case $opt in
	h) echo -e $usage
	   exit
	   ;;
	d) dir="$OPTARG"
	   ;;
	a) astarjar="$OPTARG"
	   ;;
	b) bias="$OPTARG"
	   ;;
	t) threads="$OPTARG"
	   ;;
	\?) echo "Invalid option -$OPTARG" >&2
	    ;;
    esac
done


# create directory and link probs files

mkdir -p $dir
ln -s "$PWD/tagger_out/tagProbs.txt" "$PWD/$dir/tagProbs.txt"
ln -s "$PWD/tagger_out/opProbs.txt" "$PWD/$dir/opProbs.txt"

# run the A star parser
astarcmd="java -Xmx8G -cp $astarjar de.up.ling.irtg.experimental.astar.Astar $dir --threads $threads --bias $bias --no-file-suffix --typelex typelex.gzip"
echo "running command:"
echo $astarcmd
eval $astarcmd


# link files to right places

ln -s "$PWD/$dir/results.txt" "$PWD/$dir/parserOut.txt"

# then evaluate results

bash scripts/eval.sh $dir