import argparse


### usage: python3 am_dep_fscore.py filename1 filename2
###
### Both files need to contain the same sentences in the same order + same format (no extra lines / comments etc). This
### is the case (hopefully) if you use the output of prepare_visualize.py
### concerning precision and recall, we treat file1 as gold and file2 as system output
### (e.g. if file1 has many edges and file2 few, precision is (potentially) high and recall low.
###
### author: JG

optparser = argparse.ArgumentParser(add_help=True,
    description="reads two amconll files and produces two conllu files (the intersection) sorted by size that can be compared with MaltEval")
optparser.add_argument("file1", type=str)
optparser.add_argument("file2", type=str)

opts = optparser.parse_args()

word_col = 1
head_col = 6
label_col = 7

total_edges_1 = 0
total_edges_2 = 0
total_ignored_1 = 0
total_ignored_2 = 0
unlabeled_matches = 0
labeled_matches = 0
ignored_edges = ["IGNORE", "ROOT"]

with open(opts.file1) as f1:
    with open(opts.file2) as f2:
        # this is not particularly stable, but we just go through line by line and assume everything matches up
        # we check that words match to catch when this goes wrong
        for line1, line2 in zip(f1, f2):
            line1 = line1.split("\t")
            line2 = line2.split("\t")
            if len(line1)>2:
                if len(line2) <3:
                    print("WARNING: empty line in file2, but nonempty line in file 1")
                else:
                    if not line1[word_col] == line2[word_col]:
                        print("WARNING: words in the two files don't match!")
                        print(line1[word_col])
                        print(line2[word_col])
                    else:
                        head1 = line1[head_col]
                        head2 = line2[head_col]
                        label1 = line1[label_col]
                        label2 = line2[label_col]
                        ignore1 = label1 in ignored_edges
                        ignore2 = label2 in ignored_edges
                        if not ignore1:
                            total_edges_1 += 1
                        else:
                            total_ignored_1 += 1
                        if not ignore2:
                            total_edges_2 += 1
                        else:
                            total_ignored_2 += 1
                        if (not ignore1) and (not ignore2):
                            if head1 == head2:
                                unlabeled_matches +=1
                                if label1 == label2:
                                    labeled_matches +=1
            else:
                if len(line2) > 2:
                    print("WARNING: empty line in file1, but nonempty line in file 2")

precision_u = unlabeled_matches/total_edges_2
precision_l = labeled_matches/total_edges_2
recall_u = unlabeled_matches/total_edges_1
recall_l = labeled_matches/total_edges_1

f_u = 2*precision_u*recall_u/(precision_u+recall_u)
f_l = 2*precision_l*recall_l/(precision_l+recall_l)

print("unlabeled F: %4.2f" % (f_u*100))
print("unlabeled P: %4.2f" % (precision_u*100))
print("unlabeled R: %4.2f" % (recall_u*100))
print()
print("labeled F: %4.2f" % (f_l*100))
print("labeled P: %4.2f" % (precision_l*100))
print("labeled R: %4.2f" % (recall_l*100))
print()
print("total edges file1: ", total_edges_1)
print("ignored edges file1: ", total_ignored_1)
print("total edges file2: ", total_edges_2)
print("ignored edges file2: ", total_ignored_2)