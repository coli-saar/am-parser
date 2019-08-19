local ud_prefix = "data/UD/corenlp/";

local MRP_AMR_SUBPATH = "clean_decomp";
local MRP_UCCA_SUBPATH = "very_first";

{
    "MRP_AMR_SUBPATH" : MRP_AMR_SUBPATH,
    "MRP_UCCA_SUBPATH" : MRP_UCCA_SUBPATH,

    "UD_banks": ["EWT","GUM","LinES","ParTUT"],
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

        "MRP-DM" : "data/MRP/DM/train/train.amconll",
        "MRP-PSD" : "data/MRP/PSD/train/train.amconll",
        "MRP-EDS" : "data/MRP/EDS/train/train.amconll",

        "MRP-AMR" : "data/MRP/AMR/"+MRP_AMR_SUBPATH+"/train/train.amconll",

        "MRP-UCCA" : "data/MRP/UCCA/"+MRP_UCCA_SUBPATH+"/train/train.amconll"
    },
    "gold_dev_data" : { #gold AM dependency trees for (a subset of) the dev data
        "DM" : "data/SemEval/2015/DM/gold-dev/gold-dev.amconll",
        "PAS" : "data/SemEval/2015/PAS/gold-dev/gold-dev.amconll",
        "PSD" : "data/SemEval/2015/PSD/gold-dev/gold-dev.amconll",
        "AMR-2015" : "data/AMR/2015/gold-dev/gold-dev.amconll",
        "AMR-2017" : "data/AMR/2017/gold-dev/gold-dev.amconll",
        "EDS" : "data/EDS/gold-dev/gold-dev.amconll",

        #UD:
        "EWT": ud_prefix+"EWT/dev/dev.amconll",
        "GUM": ud_prefix+"GUM/dev/dev.amconll",
        "LinES": ud_prefix+"LinES/dev/dev.amconll",
        "ParTUT": ud_prefix+"ParTUT/dev/dev.amconll",

        "MRP-DM" : "data/MRP/DM/gold-dev/gold-dev.amconll",
        "MRP-PSD" : "data/MRP/PSD/gold-dev/gold-dev.amconll",
        "MRP-EDS" : "data/MRP/EDS/gold-dev/gold-dev.amconll",

        "MRP-AMR" : "data/MRP/AMR/"+MRP_AMR_SUBPATH+"/gold-dev/gold-dev.amconll",

        "MRP-UCCA" : "data/MRP/UCCA/"+MRP_UCCA_SUBPATH+"/gold-dev/gold-dev.amconll"
    }
}

