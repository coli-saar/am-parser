local data_paths = import 'data_paths.libsonnet';

{
    "AMR-2015" : {
        "sublexica" : {
                    "edge_labels" : "data/AMR/2015/lexicon/edges.txt",
                    "constants" : "data/AMR/2015/lexicon/constants.txt",
                    "term_types" : "data/AMR/2015/lexicon/types.txt",
                    "lex_labels" : "data/AMR/2015/lexicon/lex_labels.txt"
             }
    },

    "AMR-2017" : {
        "sublexica" : {
                    "edge_labels" : "data/AMR/2017/lexicon/edges.txt",
                    "constants" : "data/AMR/2017/lexicon/constants.txt",
                    "term_types" : "data/AMR/2017/lexicon/types.txt",
                    "lex_labels" : "data/AMR/2017/lexicon/lex_labels.txt"
             }
    },

        "AMR-2020" : {
            "sublexica" : {
                        "edge_labels" : "data/AMR/2020/lexicon/edges.txt",
                        "constants" : "data/AMR/2020/lexicon/constants.txt",
                        "term_types" : "data/AMR/2020/lexicon/types.txt",
                        "lex_labels" : "data/AMR/2020/lexicon/lex_labels.txt"
                 }
        },

    "DM" : {
        "sublexica" : {
                    "edge_labels" : data_paths["SDP_prefix"]+"DM/lexicon/edges.txt",
                    "constants" : data_paths["SDP_prefix"]+"DM/lexicon/constants.txt",
                    "term_types" : data_paths["SDP_prefix"]+"DM/lexicon/types.txt",
                    "lex_labels" : data_paths["SDP_prefix"]+"DM/lexicon/lex_labels.txt"
             }
    },

    "PAS" : {
        "sublexica" : {
                    "edge_labels" : data_paths["SDP_prefix"]+"PAS/lexicon/edges.txt",
                    "constants" : data_paths["SDP_prefix"]+"PAS/lexicon/constants.txt",
                    "term_types" : data_paths["SDP_prefix"]+"PAS/lexicon/types.txt",
                    "lex_labels" : data_paths["SDP_prefix"]+"PAS/lexicon/lex_labels.txt"
             }
    },

    "PSD" : {
        "sublexica" : {
                    "edge_labels" : data_paths["SDP_prefix"]+"PSD/lexicon/edges.txt",
                    "constants" : data_paths["SDP_prefix"]+"PSD/lexicon/constants.txt",
                    "term_types" : data_paths["SDP_prefix"]+"PSD/lexicon/types.txt",
                    "lex_labels" : data_paths["SDP_prefix"]+"PSD/lexicon/lex_labels.txt"
             }
    },

    "EDS" : {
        "sublexica" : {
                    "edge_labels" : "data/EDS/lexicon/edges.txt",
                    "constants" : "data/EDS/lexicon/constants.txt",
                    "term_types" : "data/EDS/lexicon/types.txt",
                    "lex_labels" : "data/EDS/lexicon/lex_labels.txt"
             }
    },

}