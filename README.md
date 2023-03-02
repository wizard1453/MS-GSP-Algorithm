# cs583-assignment-01
Course assignment 1 for cs583 Datamining and Text Mining, uic.  
Team member: Huiyang Zhao, Yimai Zhao  

### Given files:
1. The data file. S, i.e. Sequences.
    Each line is a sequence and each item is represented with an integer number, e.g.,
    <{10, 40, 50}{40, 90}>
    <{10}{10, 40}{40}>
2. The parameter file. MS, i.e. Minimal Supports.
    It contains the MIS values of all items that have appeared in the data file: 
    MIS(1) = 0.02
    MIS(2) = 0.04
    ...
    MIS(rest) = 0.01 
    SDC = 0.03
    rest: all other items that are not given specific MIS values. SDC: support difference constraint.

### Variables:
S: Sequence. Read from given files.  
MS: Minimal support. Read from given files.  
I:  An item set of all single item. Prepare from given files.  
M:  Sorted Item set. Generated by sorting I.  
L:  Seeds generated from M.  
    A map: <item, count>
F1: Frequent 1-sequence, obtained from L.  
Ck: Candidate sequences of size k.  
F_minus1:  Frequent k-1 sequences, will be used in prune step.  
F_k: Frequent k-sequences, obtained from Ck.  
F_results:  Final itemsets with all Fk.  

### Functions:
ms_gsp:   
M <- sort(I, MS): sort the items in ascending order according to their MIS values stored in MS.  
L <- init_pass(M, S): make the first pass over the sequence data, producing the seeds set L for generating C2.  
C2 <- level2_candidate_gen_spm: designed based on level2-candidate-gen in MS-Apriori and the join step in Fig. 2.13.  
Ck <- mscandidate_gen_spm: generate Fk basing on Ck, k>2.  