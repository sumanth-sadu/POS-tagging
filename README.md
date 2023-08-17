# POS-tagging
Hands-on experience on using HMMs on part-of- speech tagging.


The .py file contains the code. The .py file and data folder containing train, dev, and test data should be in the same folder to run the code. 


Code description, brief solution and metrics are shown in .ipynb PDF


Answers to questions:

1. Selected threshold for unknown words replacement : 3
2. Total size of vocabulary : 43194 (including "< unk >" tag)
3. Total occurrences of special token '< unk >' after replacement : 32537
4. Number of transition parameters : 1378
5. Number of emission parameters : 50285
6. Accuracy of Greedy algorithm on Dev set : 93.783 %
7. Accuracy of Viterbi algorithm on Dev set : 94.494 %


Submitting 
vocab.txt - containing unique words whose occurrences are greater than 3 and < unk > tag sorted in decreasing order of their frequencies.
hmm.json - containing transition and emission probabilities. 
greedy.out - output of test data using greedy algorithm
viterbi.out -  output of test data using viterbi algorithm
