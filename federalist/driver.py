"""
This script drives the classifier script to use test each file in the test set.
"""
from __future__ import print_function
import classifier
import os
import sys

test_dir = "." + os.path.sep + "tests" + os.path.sep
class_dir = "." + os.path.sep + "training" + os.path.sep

labels = {'ham': 1, 'jay': 2, 'mad': 3}
hamilton_essays = {18, 19, 20}
jay_essays = {64}
mad_essays = {18, 19, 20, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58}

def lookup_label(testpath):
    for num in hamilton_essays:
        if str(num) in testpath:
            return labels['ham']
    for num in jay_essays:
        if str(num) in testpath:
            return labels['jay']
    for num in mad_essays:
        if str(num) in testpath:
            return labels['mad']
    assert False, "Could not figure out the label for this essay: " + testpath

def classify(testpath):
    label = lookup_label(testpath)
    trainingpaths = [os.path.join(class_dir, path) for path in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, path))]
    predicted = classifier.classify(trainingpaths, testpath)
    return predicted, label

def print_results(label, allresults):
    # TODO: Print this class's info out of the results
    # allresults is a list of tuples of the form (predicted, label)
    if label == 1:
        name = "HAMILTON"
    elif label == 2:
        name = "JAY"
    else:
        name = "MADISON"
    print("-----------INDEX:", label, name, "------------")
    classresults = [r for r in allresults if r[1] == label]
    successes = [r for r in classresults if r[0] == r[1]]
    print("Number of tests:", len(classresults))
    print("Number correctly classified:", len(successes))
    print("Precision:", len(successes) / len([r for r in allresults if r[0] == label]))
    print("Recall:", len(successes) / len(classresults))

print("Using test dir:", test_dir)
print("Using training dir:", class_dir)
results = [classify(os.path.join(test_dir, testpath)) for testpath in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir,testpath))]
labels = sorted(set((r[1] for r in results)))

for label in labels:
    print_results(label, results)
print("------------------------------------")
total_correct = [r for r in results if r[0] == r[1]]
percent = len(total_correct) / len(results) * 100
print("Percent correct:", str(percent) + "%")

