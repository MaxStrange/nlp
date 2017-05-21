"""
This module provides an interface for analyzing messages with the Semantic Analyzer,
the Politeness Analyzer, and various other ways of getting metadata.
"""
import _pickle
import atexit
from external.polite.features.vectorizer import PolitenessFeatureVectorizer
from external.polite.request_utils import check_is_request
from external.polite.scripts.format_input import format_doc
import itertools
import nltk
import numpy as np
import os
from pycorenlp import StanfordCoreNLP
import scipy
from scipy.sparse import csr_matrix
import shutil
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

# TODO: Can't get the server to behave reliably like this. So just run the server in a separate window for now.

# Start a link to the server
corenlp = StanfordCoreNLP("http://localhost:9000")

def _preprocess(msg):
    return msg.strip().strip("\"")

def analyze_message(msg):
    """
    Returns a dict of the form:
    (nwords, nsentences, nrequests, politeness, sentiment, lexicon_words, frequent_words)
    """
    msg = _preprocess(msg)
    reqs = get_requests(msg)
    politenesses = [get_politeness(r) for r in reqs] if reqs else [get_politeness(msg)]
    politeness = np.mean([item[1]['polite'] for item in list(itertools.chain.from_iterable(politenesses))])
    sentiments = get_sentiment(msg)
    sentiment = {
                    "positive": len([s for s in sentiments if int(s[2]) > 2]),
                    "neutral":  len([s for s in sentiments if int(s[2]) == 2]),
                    "negative": len([s for s in sentiments if int(s[2]) < 2])
                }
    lexwords = None # {"allsubj": ..., "disc_expansion": ..., "disc_comparison": ..., "disc_temporal_future": ..., "premise": ...}
    freqwords = None
    return {
                "n_words": len(get_words(msg)),
                "n_sentences": len(get_sentences(msg)),
                "n_requests": len(get_requests(msg)),
                "politeness": politeness,
                "sentiment": sentiment,
                "lexicon_words": lexwords,
                "frequent_words": freqwords
            }

def get_lexwords(raw_text):
    """
    Takes a string of raw text and builds a dict of discourse sense tags to words:

    {"disc_expansion": [...], "disc_comparison": [...], ...}

    Returns this dict.
    """
    with open("external/pdtb-parser/diplomacy/tmp.txt", 'w') as f:
        f.write(raw_text)
    p = subprocess.Popen(["java", "-jar", "parser.jar", "diplomacy"], cwd="external/pdtb-parser", stdout=subprocess.DEVNULL)
    p.wait(timeout=5)
    output_path = "external/pdtb-parser/diplomacy/output/tmp.txt.pipe"
    with open(output_path) as f:
        lines = [line for line in f]
    ret = "".join(lines)
    shutil.rmtree("external/pdtb-parser/diplomacy/output")
    return ret

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
        accumulated_probs.append((request['sentences'][0], probs))
    return accumulated_probs

def get_requests(raw_text):
    """
    Returns the sentences in the raw_text that are requests.
    """
    accumulated = []
    for s in get_sentences(raw_text):
        for f in format_doc(s):
            if check_is_request(f):
                accumulated.append(s)
    accumulated = list(set(accumulated))
    return accumulated

def get_sentences(raw_text):
    """
    Returns raw_text as a list of sentences.
    """
    return ["".join(s['sentences']) for s in format_doc(raw_text)]

def get_sentiment(raw_text):
    """
    Takes a string of raw text and determines the sentiment value for each sentence in it.
    """
    res = corenlp.annotate(raw_text, properties={'annotators': 'sentiment', 'outputFormat': 'json', 'timeout':1000})
    sentences = get_sentences(raw_text)
    accumulated_sents = []
    for i, s in enumerate(res['sentences']):
        try:
            accumulated_sents.append((s['index'], sentences[i], s['sentimentValue'], s['sentiment']))
        except IndexError:
            pass
    return accumulated_sents

def get_words(raw_text):
    """
    Tokenizes the given text into words and returns a list of the words.
    """
    return nltk.word_tokenize(raw_text)


if __name__ == "__main__":
    text = """
Please, if you wouldn't mind, could you get me the butter? It's across the table from you.
I know this is a pain, but I was hoping you could maybe get me the thingy. Also, I
personally believe you are a loser, and I wish you would shut up and give me the thing.
Swine, give me what I desire!"""
    print(":::::::::::::  PLANNING :::::::::::::::")
    planning = get_lexwords(text)
    print(planning)

    print("")
    print("==============================================================")
    print("")

    print(":::::::::::::  POLITENESS  :::::::::::::::")
    pol = get_politeness(text)
    print(pol)

    print("")
    print("==============================================================")
    print("")

    print(":::::::::::::  SENTIMENT  :::::::::::::::")
    sent = get_sentiment(text)
    print(sent)

    print("")
    print("==============================================================")
    print("")

    print(":::::::::::::  REQUESTS  :::::::::::::::")
    requests = get_requests(text)
    print(requests)

    print("")
    print("==============================================================")
    print("")

    print(":::::::::::::  SENTENCES  :::::::::::::::")
    sentences = get_sentences(text)
    print(sentences)

    print("")
    print("==============================================================")
    print("")


    print(":::::::::::::  WORDS  :::::::::::::::")
    words = get_words(text)
    print(words)

    print("")
    print("==============================================================")
    print("")

