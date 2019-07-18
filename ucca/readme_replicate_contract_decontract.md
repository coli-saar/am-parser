# Files to obtain the decontracted edges in mrp format using ucca scripts:

To replicate the whole pipeline of how we obtained our decontracted ucca graph in mrp format:

- first, we obtain the tokenization from the companion data (we also normalize):

 ```bash
 python3 get_companion_tokenization.py companion/data/dir companion/tokenization/dir
 ```

- Then, we use the convert_training_data_into_alto_corpus using the tokenizations we obtain from the previous step along with the mrp data (which may take a few minutes)

```bash
python3 convert_training_data_into_alto_corpus.py mrp/data/directory companion/tokenization/dir outdir/
```

- This will output two files written in the alto format, training.txt and test.txt in outdir.

- Once we run CreateCorpus from am-tools, we obtain an am-conll file which is then formatted as a .mrp file (the code that does this is not in this folder)

- With that mrp file, we run:

```bash
python3 decompress_mrp.py mrp/with/contracted/edges outdir/
```
