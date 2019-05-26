local ud_prefix = "data/UD/corenlp/";

{
    "train_data" : {
        "DM" : "data/SemEval/2015/DM/train/train.amconll",
        "PAS" : "data/SemEval/2015/PAS/train/train.amconll",
        "PSD" : "data/SemEval/2015/PSD/train/train.amconll",
        "AMR-2015" : "data/AMR/2015/train/train.amconll",
        "AMR-2017" : "data/AMR/2017/train/train.amconll",
        "EDS" : "data/EDS/train/train.amconll",
        #UD:
        "EWT": ud_prefix+"EWT/train/train.amconll",
        "GUM": ud_prefix+"GUM/train/train.amconll",
        "LinES": ud_prefix+"LinES/train/train.amconll",
        "ParTUT": ud_prefix+"ParTUT/train/train.amconll",
    },
    "gold_dev_data" : { #gold AM dependency trees for (a subset of) the dev data
        "DM" : "data/SemEval/2015/DM/gold-dev/gold-dev.amconll",
        "PAS" : "data/SemEval/2015/PAS/gold-dev/gold-dev.amconll",
        "PSD" : "data/SemEval/2015/PSD/gold-dev/gold-dev.amconll",
        "AMR-2015" : "data/AMR/2015/gold-dev/gold-dev.amconll",
        "AMR-2017" : "data/AMR/2015/gold-dev/gold-dev.amconll", #that one is the same dev set as AMR-2015.
        "EDS" : "data/EDS/gold-dev/gold-dev.amconll",

        #UD:
        "EWT": ud_prefix+"EWT/dev/dev.amconll",
        "GUM": ud_prefix+"GUM/dev/dev.amconll",
        "LinES": ud_prefix+"LinES/dev/dev.amconll",
        "ParTUT": ud_prefix+"ParTUT/dev/dev.amconll",
    }
}

