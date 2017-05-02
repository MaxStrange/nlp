"""
This module provides utility functions for yielding the data
in various ways.
"""
from   collections import namedtuple
import json
import numpy as np
import os
import random

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
    def __init__(self, data, last):
        self.year = int(data["season"])
        self.season = "spring" if (data["season"] - self.year) == 0 else "winter"
        self.interaction = data["interaction"]

        MessagePair = namedtuple("MessagePair", ["betrayer", "victim"])
        betrayer_messages = data["messages"]["betrayer"]
        victim_messages = data["messages"]["victim"]
        zipped = zip(betrayer_messages, victim_messages)
        self.messages = [MessagePair(betrayer=Message(m[0]), victim=Message(m[1])) for m in zipped]
        self.is_last_season_in_relationship = last

    def __str__(self):
        s  = "Year: " + str(self.year) + " "
        s += "Season: " + str(self.season) + " "
        s += "Interaction: " + str(self.interaction) + " "
        s += "Number of messages: " + str(len(self.messages))
        return s

    def _avg_betrayer(self, attr):
        """
        Gets the average value of all messages for the given attribute in this Season.
        """
        l = [getattr(m, attr) for m in self.get_messages('betrayer')]
        return sum(l) / len(l) if l else 0.0

    def _avg_victim(self, attr):
        """
        Gets the average value of all messages for the given attribute in this Season.
        """
        l = [getattr(m, attr) for m in self.get_messages('victim')]
        return sum(l) / len(l) if l else 0.0

    def get_messages(self, person='betrayer'):
        """
        Generator for all the Message objects in this object.
        """
        for mp in self.messages:
            if person == "betrayer":
                yield mp.betrayer
            else:
                yield mp.victim

    def to_feature_vector(self, reverse):
        """
        Returns a feature vector version of this Season.
        """
        b = [self._avg_betrayer('nwords'), self._avg_betrayer('nsentences'), self._avg_betrayer('nrequests'),
             self._avg_betrayer('politeness'), self._avg_betrayer('avg_sentiment')]
        v = [self._avg_victim('nwords'), self._avg_victim('nsentences'), self._avg_victim('nrequests'),
             self._avg_victim('politeness'), self._avg_victim('avg_sentiment')]
        if reverse:
            return v + b
        else:
            return b + v


class Relationship:
    """
    An instance of one of the 500 relationship sequences.
    """
    def __init__(self, data):
        self.idx = data["idx"]              # Unique ID as a dataset entry
        self.game = data["game"]            # Unique ID of the game this sequence comes from
        self.betrayal = data["betrayal"]    # Whether the relationship ended in betrayal
        self.people = data["people"]        # The countries represented by the two players
        self.seasons = [Season(s, False) for s in data["seasons"]]
        self.seasons[-1].is_last_season_in_relationship = True

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

def _concatenate_trigram(tri, reverse):
    """
    Takes an iterable of three Season objects and returns a numpy vector of
    all the features of each Season concatenated.
    """
    fv = []
    for s in tri:
        fv += s.to_feature_vector(reverse)
    return np.array(fv)

def get_X_feed():
    """
    Generator for getting all the X vectors. Returns a tuple of (reversed, X) at each yield.

    Each X is a numpy array that looks like this:
    [N_words_Betrayer_season0, N_sentences_Betrayer_season0, N_requests_season0, politeness_season0, sentiment_season0, N_words_Victim_season0, ..., N_words_Victim_season1, ..., sentiment_season2]

    That is:
    [B, B, B, B, B, V, V, V, V, V          <- Season 0
     B, B, B, B, B, V, V, V, V, V          <- Season 1
     B, B, B, B, B, V, V, V, V, V]         <- Season 2

    When reverse is True, we get:
    [V, V, V, V, V, B, B, B, B, B
     V, V, V, V, V, B, B, B, B, B
     V, V, V, V, V, B, B, B, B, B]
    """
    for relationship in get_all_sequences():
        season_trigrams = relationship.get_season_trigrams()
        for tri in season_trigrams:
            reverse = random.choice([True, False])
            yield reverse, _concatenate_trigram(tri, reverse)

def _get_label_from_trigram(tri, betrayal, reverse):
    """
    Gets the Y corresponding to the given trigram.

    If reverse is True, then the returned label vector is:
    [Victim, Victim, Victim, B, B, B] rather than [B, B, B, V, V, V]
    """
    def finish_with_ones(acc, reverse):
        acc += [1, 0, 0, 0]
        if reverse:
            return acc[3:] + acc[:3]
        else:
            return acc

    if not betrayal:
        # labels are all zeros - nobody ever betrays anyone
        return np.array([0, 0, 0, 0, 0, 0])
    else:
        acc = []
        for i, s in enumerate(tri):
            if i == 2:
                # If this is the last value in the trigram, it may be the last season in the relationship
                if s.is_last_season_in_relationship:
                    # If this is the last season in the relationship, it is when the betrayal started.
                    # All betrayal labels for the next turns are 1.
                    return np.array(finish_with_ones(acc, reverse))
                else:
                    acc.append(0)
                    # No betrayal yet
                    return np.array(acc + [0, 0, 0])
            else:
                acc.append(0)

def get_Y_feed(X):
    """
    Generator for getting each Y vector (label vector) that corresponds to each X vector.
    X is a list of: [(reversed, fv), (reversed, fv), ...]

    A Y vector is a numpy array that looks like this:
    [A_betrays_B_in_one_season, A_betrays_B_in_two_seasons, A_betrays_B_in_three_seasons, B_betrays_A_in_one_season, B_betrays_A_in_two_seasons, B_betrays_A_in_three_seasons]

    IMPORTANT:
    As soon as someone betrays the other, the odds collapse to 1 and 0, so that if A betrays B in two seasons, then the vector will look like this:
    [0, 1, 1,
     0, 0, 0]
    """
    for i, relationship in enumerate(get_all_sequences()):
        season_trigrams = relationship.get_season_trigrams()
        for tri in season_trigrams:
            yield _get_label_from_trigram(tri, relationship.betrayal, X[i][0])


if __name__ == "__main__":
    # Debug
    print("Getting data...")
    data = [d for d in get_all_sequences()]

    print("Length of data:", len(data))

    betrayals = [d for d in data if d.betrayal]
    print("Number of sequences that end in betrayal:", len(betrayals))

    Xs = [x for x in get_X_feed()]
    Ys = [y for y in get_Y_feed(Xs)]
    for y in Ys:
        print(y)

#    print("Data0<<:", data[0], ">>")
#    print("Season0:<<", data[0].seasons[0], ">>")
#    print("MessagePair0:<<", data[0].seasons[0].messages[0], ">>")

#    lengths = [len(seq.seasons) for seq in data]
#    sorted_lengths = sorted(lengths)
#    print(sorted_lengths)

