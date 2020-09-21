local ud_prefix = "data/UD/corenlp/";

local MRP_AMR_SUBPATH = "clean_decomp";
local MRP_UCCA_SUBPATH = "very_first";

local SDP_prefix = "data/SemEval/2015/";
local my_prefix = "data/ACL2019/";
{
    "GLOVE_DIR" : "/local/mlinde/glove/",

    "MRP_AMR_SUBPATH" : MRP_AMR_SUBPATH,
    "MRP_UCCA_SUBPATH" : MRP_UCCA_SUBPATH,
    "SDP_prefix" : SDP_prefix,

    "UD_banks": ["EWT","GUM","LinES","ParTUT"],
    "train_data" : {
        "DM" : my_prefix+"SemEval/2015/DM/train/train_small.amconll", 
#        "DM" : my_prefix+"DM/output/train/train.amconll",
        "PAS" : my_prefix+"SemEval/2015/PAS/train/train.amconll",
        "PSD" : my_prefix+"SemEval/2015/PSD/train/train.amconll",
        "AMR-2015" : my_prefix+"AMR/2015/train/train.amconll",
        "AMR-2017" : my_prefix+"AMR/2017/train/train.amconll",
        "EDS" : my_prefix+"EDS/train/train.amconll",
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
        "DM" : my_prefix+"SemEval/2015/DM/gold-dev/gold-dev.amconll",
#        "DM" : my_prefix+"DM/output/gold-dev/gold-dev.amconll",
        "PAS" : my_prefix+"SemEval/2015/PAS/gold-dev/gold-dev.amconll",
        "PSD" : my_prefix+"SemEval/2015/PSD/gold-dev/gold-dev.amconll",
        "AMR-2015" : my_prefix+"AMR/2015/gold-dev/gold-dev.amconll",
        "AMR-2017" : my_prefix+"AMR/2017/gold-dev/gold-dev.amconll",
        "EDS" : my_prefix+"EDS/gold-dev/gold-dev.amconll",

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

