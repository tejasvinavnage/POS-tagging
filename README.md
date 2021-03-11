# POS-tagging
Part Of Speech Tagging NLP Python program

The program tagger.py will take as input a training file containing part of speech tagged text, and a file containing text to be part of speech tagged. 
The program implements the "most likely tag" baseline.

Assumption: Any word found in the test data but not in training data (i.e. an unknown word) is an NN.

The training data is pos-train.txt, and the text to be tagged is pos-test.txt. 
There is also a gold standard (manually tagged) version of the test file found in pos-test-key.txt that is used to evaluate the tagged output.
