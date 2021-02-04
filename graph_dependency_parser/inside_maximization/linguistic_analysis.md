# Linguistic Analysis
To be added to the wiki when we publish the paper

## Counting and printing examples of phenomena

There are three Java scripts that categorise corpus entries, count the entries in each category, and print new amconll files for each category. These are:
  1. `CountEdges`: creates a map from AM operation (= dep tree edge labels) to sentences that use that operation. 
   * Creates a directory called `edges` and a daughter `edges/examples`
   * Prints a summary of the counts to `edges/summary.txt`, e.g. 
   
   > Edges in train-final/DM
   >
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
      > 0. ARG1  ####  3
      > S2: 197467
      > S0: 12288
      > S1: 3403
      >
      > ...
    * For each (edge, source) pair, prints a file of all sentences containing a word whose supertag has an edge incident to that source. Files are named after the edge and source, e.g. `ARG1_s0` and are located in `examples/`.
    
    
    
