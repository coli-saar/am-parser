#!/bin/bash




usage="Preprocess the corpus.\n\n

Arguments: \n
\n\t     -i  directory which contains all the test corpus files (e.g. data/amrs/split/test in the official AMR corpora)
\n\t     -o  output directory
\n\t     -j  am-tools jar file
"

while getopts "i:o:j:h" opt; do
    case $opt in
	h) echo -e $usage
	   exit
	   ;;
	i) input="$OPTARG"
	   ;;
	o) outputPath="$OPTARG"
	   ;;
	j) jar="$OPTARG"
	   ;;
    \?) echo "Invalid option -$OPTARG" >&2
	   ;;
    esac
done

if [ "$input" = "" ]; then
    printf "\n No input directory given. Please use -i option."
    exit 1
fi


if [ "$outputPath" = "" ]; then
    printf "\n No output directory given. Please use -o option."
    exit 1
fi

if [ "$jar" = "" ]; then
    printf "\n No jar file given. Please use -j option."
    exit 1
fi



mkdir -p $outputPath
log=$outputPath/preprocess.log
if [ -f "$log" ]; then
    rm "$log"
fi

testNNdata=$outputPath/nnData/test/
testAltodata=$outputPath/alto/test/

# a lot of scripts live here so let's store it as a variable in case it changes
datascriptPrefix="de.saar.coli.amrtagging.formalisms.amr.tools.datascript"


# make subdirectories in the output directory
mkdir -p $testNNdata  # NN test set 
# for alto to read
mkdir -p $testAltodata             # test input for evaluation


NNdataCorpusName="namesDatesNumbers_AlsFixed_sorted.corpus"  # from which we get the NN training data
evalDataCorpusName="finalAlto.corpus"                        # from which we get the dev and test evaluation data
trainMinuteLimit=600                                         # limit for generating NN training data
devMinuteLimit=20                                            # limit for geneating NN dev data
threads=1
memLimit=6G
posTagger="downloaded_models/stanford/english-bidirectional-distsim.tagger"
nerTagger="downloaded_models/stanford/english.conll.4class.distsim.crf.ser.gz"
wordnet="downloaded_models/wordnet3.0/dict/"

# disable use of conceptnet by replacing this with 'conceptnet=""'
#conceptnet="--conceptnet resources/conceptnet-assertions-5.7.0.csv.gz"
conceptnet=""


# Raw dev and test to Alto format

# test set
testRawCMD="java -Xmx$memLimit -cp $jar $datascriptPrefix.FullProcess --amrcorpus $input --output $testAltodata  >>$log 2>&1"
printf "\nconverting test set to Alto format for evaluation\n"
printf "\nconverting test set to Alto format for evaluation\n" >> $log
echo $testRawCMD >> $log
eval $testRawCMD

# test eval input data preprocessing
testEvalCMD="java -Xmx$memLimit -cp $jar $datascriptPrefix.MakeDevData -c $testAltodata -o $testNNdata --stanford-ner-model $nerTagger --tagger-model $posTagger >>$log 2>&1"
printf "\ngenerating evaluation input (full corpus) for test data\n"
printf "\ngenerating evaluation input (full corpus) for test data\n" >> $log
echo $testEvalCMD  >> $log
eval $testEvalCMD

# move some stuff around
printf "\nMoving some files around. If you get errors here, something above didn't work. Check the logfile in $log\n"
cp $testAltodata/raw.amr $testNNdata/goldAMR.txt


#Create empty amconll for test set
emptyTestAmconllCMD="java -Xmx$memLimit -cp $jar de.saar.coli.amrtagging.formalisms.amr.tools.PrepareTestDataFromFiles -c $testNNdata -o $outputPath --prefix test --stanford-ner-model $nerTagger >>$log 2>&1"
printf "\nGenerate empty amconll test data\n"
eval $emptyTestAmconllCMD

printf "\neverything is in $outputPath\n"
