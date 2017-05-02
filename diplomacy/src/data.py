"""
This module provides utility functions for yielding the data
in various ways.
"""
from   collections import namedtuple
import json
import numpy as np
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
        if len(self.sentiment) == 0:
            self.avg_sentiment = 0.0
        else:
            self.avg_sentiment = sum(list(self.sentiment.values())) / len(list(self.sentiment.values()))

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

    def _avg(self, attr):
        """
        Gets the average value of all messages for the given attribute in this Season.
        """
        l = [getattr(m, attr) for m in self.get_messages()]
        return sum(l) / len(l) if l else 0.0

    def get_messages(self):
        """
        Generator for all the Message objects in this object.
        """
        for mp in self.messages:
            yield mp.betrayer
            yield mp.victim

    def to_feature_vector(self):
        """
        Returns a feature vector version of this Season.
        """
        return [self._avg('nwords'), self._avg('nsentences'), self._avg('nrequests'),
                self._avg('politeness'), self._avg('avg_sentiment')]


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

    def get_season_trigrams(self):
        """
        Returns a list of trigram seasons. If a relationship is five seasons long:
        [S01, W01, S02, W02, S03] -> [(S01, W01, S02), (W01, S02, W02), (S02, W02, S03)]
        """
        trigrams = []
        for i in range(len(self.seasons)):
            if i + 2 < len(self.seasons):
                trigrams.append((self.seasons[i], self.seasons[i + 1], self.seasons[i + 2]))
        return trigrams


def get_all_sequences():
    """
    Gets all the relationship sequences and yields them one at a time.
    """
    with open(data_path) as f:
        data = json.load(f)

    for seq in data:
        yield Relationship(seq)

def _concatenate_trigram(tri):
    """
    Takes an iterable of three Season objects and returns a numpy vector of
    all the features of each Season concatenated.
    """
    fv = []
    for s in tri:
        fv += s.to_feature_vector()
    return np.array(fv)

def get_X_feed():
    """
    Generator for getting all the X vectors.

    Each X is a numpy array that looks like this:
    [N_words_season0, N_sentences_season0, N_requests_season0, politeness_season0, sentiment_season0, N_words_season1, ..., N_words_season2, ..., sentiment_season2]
    """
    for relationship in get_all_sequences():
        season_trigrams = relationship.get_season_trigrams()
        for tri in season_trigrams:
            yield _concatenate_trigram(tri)

def _get_label_from_trigram(tri, betrayal):
    """
    Gets the Y corresponding to the given trigram.
    """
    if not betrayal:
        # labels are all zeros - nobody ever betrays anyone
        return np.array([0, 0, 0, 0, 0, 0])
    else:
        for s in tri:

def get_Y_feed():
    """
    Generator for getting each Y vector (label vector) that corresponds to each X vector.

    A Y vector is a numpy array that looks like this:
    [A_betrays_B_in_one_season, A_betrays_B_in_two_seasons, A_betrays_B_in_three_seasons, B_betrays_A_in_one_season, B_betrays_A_in_two_seasons, B_betrays_A_in_three_seasons]

    IMPORTANT:
    As soon as someone betrays the other, the odds collapse to 1 and 0, so that if A betrays B in two seasons, then the vector will look like this:
    [0, 1, 1,
     0, 0, 0]
    """
    for relationship in get_all_sequences():
        season_trigrams = relationship.get_season_trigrams()
        for tri in season_trigrams:
            yield _get_label_from_trigram(tri, relationship.betrayal)


if __name__ == "__main__":
    # Debug
    print("Getting data...")
    data = [d for d in get_all_sequences()]

    print("Length of data:", len(data))

    betrayals = [d for d in data if d.betrayal]
    print("Number of sequences that end in betrayal:", len(betrayals))

    print("Season.interactions:")
    for r in data:
        for s in r.seasons:
            print(s.interaction)
        print("Ended in betrayal?", r.betrayal)
        print("====================================")

#    print("Data0<<:", data[0], ">>")
#    print("Season0:<<", data[0].seasons[0], ">>")
#    print("MessagePair0:<<", data[0].seasons[0].messages[0], ">>")

#    lengths = [len(seq.seasons) for seq in data]
#    sorted_lengths = sorted(lengths)
#    print(sorted_lengths)

