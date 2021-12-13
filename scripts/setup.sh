#/bin/bash

git submodule init
git submodule update

pushd evaluation_tools/fast_smatch
echo "Building fast_smatch (for evaluation of EDS)"
bash build.sh
popd

mkdir -p tmp
wget http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz -O tmp/WordNet-3.0.tar.gz
tar -xzvf tmp/WordNet-3.0.tar.gz -C tmp
mkdir -p downloaded_models/wordnet3.0
mv tmp/WordNet-3.0/dict downloaded_models/wordnet3.0/
rm -rf tmp


jar=am-tools.jar

if [ -f "$jar" ]; then
    echo "jar file found at $jar"
else
    echo "jar file not found at $jar, downloading it!"
    wget -O "$jar" http://www.coli.uni-saarland.de/projects/amparser/am-tools.jar
fi



mkdir -p data/
mkdir -p models/

#Because of naming conventions, the EDS config file uses a "non-existent gold file" gold-dev/gold-dev
#Create an empty dummy here, in reality, we will open gold-dev/gold-dev.amr.txt and gold-dev/gold-dev.edm
mkdir -p data/EDS/dev/
mkdir -p data/EDS/test/
touch data/EDS/dev/dev-gold
touch data/EDS/test/test-gold


