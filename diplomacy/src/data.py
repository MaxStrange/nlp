"""
This module provides utility functions for yielding the data
in various ways.
"""
from   collections import namedtuple
import json
import os

data_path = os.path.join("..", "data_from_paper", "diplomacy_data.json")

class Message:
    """
    A message is a message sent either from the betrayer or from the victim.
    It does NOT contain the raw text when the data comes from the training data.
    """
    def __init__(self, data):
        self.nwords = data["n_words"]
        self.nsentences = data["n_sentences"]
        self.nrequests = data["n_requests"]
        self.politeness = data["politeness"]
        self.sentiment = data["sentiment"]
        self.lexicon_words = data["lexicon_words"]
        self.frequent_words = data["frequent_words"]

    def __str__(self):
        s  = "n words: " + str(self.nwords) + " "
        s += "n sentences: " + str(self.nsentences) + " "
        s += "n requests: " + str(self.nrequests) + " "
        s += "politeness: " + str(self.politeness) + " "
        s += "sentiment: " + str(self.sentiment) + " "
        s += "lexicon_words: " + str(self.lexicon_words) + " "
        s += "frequent_words: " + str(self.frequent_words) + " "
        return s


class Season:
    """
    A game turn along with all the messages sent during that time.
    """
    def __init__(self, data):
        self.year = int(data["season"])
        self.season = "spring" if (data["season"] - self.year) == 0 else "winter"
        self.interaction = data["interaction"]

        MessagePair = namedtuple("MessagePair", ["betrayer", "victim"])
        betrayer_messages = data["messages"]["betrayer"]
        victim_messages = data["messages"]["victim"]
        zipped = zip(betrayer_messages, victim_messages)
        self.messages = [MessagePair(betrayer=Message(m[0]), victim=Message(m[1])) for m in zipped]

    def __str__(self):
        s  = "Year: " + str(self.year) + " "
        s += "Season: " + str(self.season) + " "
        s += "Interaction: " + str(self.interaction) + " "
        s += "Number of messages: " + str(len(self.messages))
        return s


class Relationship:
    """
    An instance of one of the 500 relationship sequences.
    """
    def __init__(self, data):
        self.idx = data["idx"]              # Unique ID as a dataset entry
        self.game = data["game"]            # Unique ID of the game this sequence comes from
        self.betrayal = data["betrayal"]    # Whether the relationship ended in betrayal
        self.people = data["people"]        # The countries represented by the two players
        self.seasons = [Season(s) for s in data["seasons"]]


    def __str__(self):
        s  = "ID: " + str(self.idx) + " "
        s += "Game: " + str(self.game) + " "
        s += "Betrayal: " + str(self.betrayal) + " "
        s += "People: " + str(self.people) + " "
        s += "Length of relationship: " + str(len(self.seasons)) + " "
        return s


def get_all_sequences():
    """
    Gets all the relationship sequences and yields them one at a time.
    """
    with open(data_path) as f:
        data = json.load(f)

    for seq in data:
        yield Relationship(seq)

if __name__ == "__main__":
    # Debug
    print("Getting data...")
    data = [d for d in get_all_sequences()]

    print(data[0].seasons[0].messages[0].betrayer.lexicon_words)

#    print("Data0<<:", data[0], ">>")
#    print("Season0:<<", data[0].seasons[0], ">>")
#    print("MessagePair0:<<", data[0].seasons[0].messages[0], ">>")

#    lengths = [len(seq.seasons) for seq in data]
#    sorted_lengths = sorted(lengths)
#    print(sorted_lengths)

