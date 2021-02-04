# Linguistic Analysis
To be added to the wiki when we publish the paper

## Counting and printing examples of phenomena

There are three Java scripts that categorise corpus entries, count the entries in each category, and print new amconll files for each category. These are:
  1. `CountEdges`: creates a map from AM operation (= dep tree edge labels) to sentences that use that operation. 
   * Creates a directory called `edges` and a daughter `edges/examples`
   * Prints a summary of the counts to `edges/summary.txt`, e.g. 
   
   > Edges in train-final/DM
   > 0. MOD_S2  ####  308231
   > 1. APP_S0  ####  171516
   > 2. IGNORE  ####  169012
   > 3. APP_S2  ####  73070
   > 4. ROOT  ####  32942
   > 5. MOD_S1  ####  6426
   > 6. APP_S1  ####  5846
   > 7. MOD_S0  ####  2647

   * Prints one amconll file for each operation containing all sentences that use it. Files are named after their edges, e.g. `MOD_S2.amconll` and are located in `examples/`.
   
  2. `CountSources`: creates a map from sources to a map from graph edge labels to sentences containing a word whose graph has that edge label incident to that source. e.g. ARG0 -> S1
   * Creates a directory called `sources` and a daughter `sources/examples`
   * Prints a summary of the counts to `sources/summary.txt`, e.g. 
   
   > Sources by graph edge label in train-final/DM
   > 
   > 0. ARG1  #### 3<br>
   > S2: 19746<br>
   > S0: 12288<br>
   > S1: 3403<br>    
   > ...
      
   * For each (edge, source) pair, prints a file of all sentences containing a word whose supertag has an edge incident to that source. Files are named after the edge and source, e.g. `ARG1_s0` and are located in `examples/`.
    
  3. `CountSupertags`: creates a map from unlabelled graph constants to supertags to sentences containing a word with that supertag
    
   * Creates a directory called `supertags` and a daughter `supertags/examples`
   * Prints a summary of the counts to `supertags/summary.txt`, e.g. 
    
   > 3. [i_4<root>/--LEX-- -ARG1-> i_5]  ####  7<br>
   > (i_4<root> / --LEX--  :ARG1 (i_5<S2>)),(S2()): 88186<br>
   > (i_4<root> / --LEX--  :ARG1 (i_5<S0>)),(S0()): 8502<br>
   > (i_4<root> / --LEX--  :ARG1 (i_5<S0>)),(S0(S2())): 120<br>
   > (i_4<root> / --LEX--  :ARG1 (i_5<S1>)),(S1()): 73<br>
   > (i_4<root> / --LEX--  :ARG1 (i_5<S1>)),(S1(S2())): 30<br>
   > (i_4<root> / --LEX--  :ARG1 (i_5<S2>)),(S2(S1())): 1<br>
   > (i_4<root> / --LEX--  :ARG1 (i_5<S0>)),(S0(S1())): 1
    
   * For each supertag, prints an amconll file of all sentences containing a word that is assigned that supertag. Because supertags are full of weird symbols, files are instead numbered. `<i>_<j>.amconll` where `<i>` is the number printed before the graph constant, e.g. `3` above, and `<j>` is the index of the supertag in the list of supertags under that graph, starting at 0. In the example above, to find the 120 sentences that contain the graph constant `[i_4<root>/--LEX-- -ARG1-> i_5]` with supertag `(i_4<root> / --LEX--  :ARG1 (i_5<S0>)),(S0(S2()))`, look in file `3_2.amconll`.
