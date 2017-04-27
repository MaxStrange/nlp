"""
This module provides an interface for analyzing messages with the Semantic Analyzer,
the Politeness Analyzer, and various other ways of getting metadata.
"""
import _pickle
import atexit
from external.polite.features.vectorizer import PolitenessFeatureVectorizer
from external.polite.scripts.format_input import format_doc
import nltk
import numpy as np
import os
import scipy
from scipy.sparse import csr_matrix
import sklearn
import subprocess
import time

## Start up the coreNLP server
#corenlp = subprocess.Popen(["java", "-mx4g", "-cp", "../external/stanford-corenlp-full-2016-10-31/*", "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", "-port", "9000", "-timeout", "1500"], stdout=subprocess.DEVNULL)
#
## Kill it when we exit
#atexit.register(corenlp.kill)

# Load up the politeness model
POLITE_FILEPATH = os.path.join(os.path.split(__file__)[0], "politeness-svm.p")
politeness_model = _pickle.load(open(POLITE_FILEPATH, 'rb'), encoding='latin1', fix_imports=True)

# Sleep a few seconds to let the server start
#time.sleep(4)

# TODO: Can't get the server to behave reliably like this. So just run the server in a separate window.

def get_politeness(raw_text):
    """
    Takes a string of raw text and determines the politeness value for it.
    Works best if the given text contains requests, rather than statements,
    but will try regardless.
    """
    vectorizer = PolitenessFeatureVectorizer()
    formatted = format_doc(raw_text)
    accumulated_probs = []
    for request in formatted:
        features = vectorizer.features(request)
        fv = [features[f] for f in sorted(features.keys())]
        X = csr_matrix(np.asarray([fv]))
        probs = politeness_model.predict_proba(X)
        probs = {"polite": probs[0][1], "impolite": probs[0][0]}
        accumulated_probs.append(probs)
    return probs


if __name__ == "__main__":
    text = """
Please, if you wouldn't mind, could you get me the butter? It's across the table from you.
I know this is a pain, but I was hoping you could maybe get me the thingy. Also, I
personally believe you are a loser, and I wish you would shut up and give me the thing.
Swine! Give me what I desire!"""
    pol = get_politeness(text)
    print(pol)

