"""
This file takes the federalist papers .txt file and separates it into one training file per author and a directory of test files.
The text file that this script takes as an input comes from:
http://www.gutenberg.org/ebooks/1404
"""
from __future__ import print_function
import os
import sys

if len(sys.argv) < 2:
    print("Usage:", sys.argv[0], "path/to/federalist.txt")
    exit(1)
else:
    fedpath = sys.argv[1]

with open(fedpath) as f:
    print("Reading all the lines in the file into memory...")
    lines = [line for line in f]

try:
    os.mkdir("training")
except FileExistsError:
    print("There is already a training directory. Putting the files in there.")
try:
    os.mkdir("tests")
except FileExistsError:
    print("There is already a tests directory. Putting the files in there.")

def essay_is_disputed(number):
    if number >= 18 and number <= 20:
        return True
    elif number >= 49 and number <= 58:
        return True
    elif number == 64:
        return True
    else:
        return False

def line_is_author(line):
    return line.strip().startswith("HAMILTON") or line.strip().startswith("JAY") or line.strip().startswith("MADISON")

def generate_data(lines):
    """
    Generator method for yielding up each essay one by one along with its number and author.
    """
    accumulated_essay = ""
    essay_started = False
    author = ""
    num = -1
    for line in lines:
        # Iterate through each line and check if it is an author, a number, or a start.
        # If it isn't any of those things and we haven't started accumulating yet, just discard it
        if line.strip().startswith("FEDERALIST"):
            if essay_started:
                author = "disputed" if essay_is_disputed(num) else author
                yield num, author, accumulated_essay.strip()
                accumulated_essay = ""
            fed, no, num = line.strip().split()
            num = int(num)
            essay_started = False
        elif line_is_author(line):
            author = line.strip().replace(' ', '_')
            essay_started = True

        if essay_started and not line_is_author(line):
            accumulated_essay += line

print("Going through each paper and saving it as either a test or a training file...")
jay = ham = mad = ""
for num, author, text in generate_data(lines):
    if author == "disputed":
        with open("tests" + os.path.sep + author + str(num) + ".txt", 'w') as f:
            f.write(text)
    else:
        if author.startswith("HAMILTON"):
            ham += text
        elif author.startswith("MADISON"):
            mad += text
        else:
            jay += text

with open("training" + os.path.sep + "jay.txt", 'w') as f:
    f.write(jay)
with open("training" + os.path.sep + "mad.txt", 'w') as f:
    f.write(mad)
with open("training" + os.path.sep + "ham.txt", 'w') as f:
    f.write(ham)

