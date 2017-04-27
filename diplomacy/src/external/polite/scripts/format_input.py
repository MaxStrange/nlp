import json
import os
import re
import requests
import sys
import traceback

from json import JSONDecodeError
from requests.exceptions import RequestException
from nltk.tokenize import sent_tokenize
from pycorenlp import StanfordCoreNLP

nlpserver = StanfordCoreNLP("http://localhost:9000")

def clean_depparse(dep):
    """
    Given a dependency dictionary, return a formatted string representation.
    """
    return str(dep['dep'] + "(" + dep['governorGloss'].lower() + "-" +
               str(dep['governor']) + ", " + dep['dependentGloss'] + "-" +
               str(dep['dependent']) + ")")

def clean_treeparse(tree):
    cleaned_tree = re.sub(r' {2,}', ' ', tree)
    cleaned_tree = re.sub(r'\n', '', cleaned_tree)
    cleaned_tree = re.sub(r'\([^\s]*\s', '', cleaned_tree)
    cleaned_tree = re.sub(r'\)', '', cleaned_tree)
    cleaned_tree = re.sub(r'-LRB-', '(', cleaned_tree)
    cleaned_tree = re.sub(r'-RRB-', ')', cleaned_tree)

    return cleaned_tree

def get_sentences(doc_text):
    return sent_tokenize(doc_text.strip().replace("\n", " "))

def get_parses(sent):
    parse = {'deps': [], 'sent': ""}
    response = nlpserver.annotate(sent, properties={'annotators': 'tokenize,ssplit,pos,parse,depparse', "outputFormat": "json"})
    for sentence in response['sentences']:
        parse['deps'] = sentence['enhancedPlusPlusDependencies']
        parse['sent'] = sent

    return parse

def format_doc(doc_text):
    print("Getting the sentences...")
    sents = get_sentences(doc_text)
    print("Parsing the sentences...")
    raw_parses = []
    for sent in sents:
        raw_parses.append(get_parses(sent))
    print("Formatting the raw parses...")
    results = []
    for raw in raw_parses:
        result = {'parses': [], 'sentences': []}
        for dep in raw['deps']:
            result['parses'].append(clean_depparse(dep))
        result['sentences'].append(clean_treeparse(raw['sent']))

        results.append(result)

    return results
