from graph_dependency_parser.components.dataset_readers.amconll_tools import AMSentence, parse_amconll

with open("../../experimentData/unsupervised2020/predictedTrainAmconlls/AMRAuto4-jan15/AMR-2017_amconll_list_dev_epoch57.amconll") as f:
    sentences = parse_amconll(f)

    for sent in sentences:
        for word in sent.words:
            if word.lexlabel == "_":
                print(str(sent))
