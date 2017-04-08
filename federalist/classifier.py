"""
This script takes three inputs:
an integer value -> the number of classes
a list of files of that number -> the training files (one for each class)
a test file -> the file you want to test for authorship classification

e.g.:
python3 classifier.py 3 ham.txt jay.txt mad.txt test.txt
>>outputs: 0 if classifier predicts Hamilton, 1 for Jay, 2 for Madison.
"""
from __future__ import print_function
import gzip
import os
import shutil
import sys

if len(sys.argv) <= 4:
    print("USAGE: n A0 A1 A2 ... An T, where n is the number of classes and the number of training files, Ai is the ith class's training file, and T is the test file you want to classify.")
    exit(1)
else:
    num_classes = int(sys.argv[1])
    test_path = sys.argv[-1]
    classifier_paths = [a for a in sys.argv[2:-1]]
    if len(classifier_paths) != num_classes:
        print("Number of files given must match the number given as the first argument.")
        exit(1)

def compress(path):
    with open(path, 'rb') as org:
        with gzip.open(path + ".gz", 'wb') as zipped:
            zipped.writelines(org)
        sz = os.path.getsize(path + ".gz")
        os.remove(path + ".gz")
    return sz

def concatenate(path, test_path):
    with open(test_path, 'r') as f:
        lines = [line for line in f]
    shutil.copyfile(path, path + "tmp.txt")
    with open(path + "tmp.txt", 'a') as p:
        for line in lines:
            p.write(line)
    sz = compress(path + "tmp.txt")
    os.remove(path + "tmp.txt")
    return sz

training_file_lengths = [compress(a) for a in classifier_paths]
print(training_file_lengths)
concatenated = [concatenate(a, test_path) for a in classifier_paths]
print(concatenated)
sizes = [c - x for c, x in zip(concatenated, training_file_lengths)]
print(sizes)
yval = min(sizes)
for index, sz in enumerate(sizes):
    if sz == yval:
        y = index
        break

print(y, "(" + classifier_paths[y] + ")")

