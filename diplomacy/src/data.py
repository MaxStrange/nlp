"""
This module provides utility functions for yielding the data
in various ways.
"""
from   collections import namedtuple
import json
import numpy as np
import os
import random
random.seed(12345)
np.random.seed(12345)

# The default data path
DATA_PATH = os.path.join("..", "data_from_paper", "diplomacy_data.json")
UPSAMPLE_TIMES = 4
validation_set = None
training_set = None
already_got_all_sequences = False

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
        try:
            self.temporal = self.lexicon_words["disc_temporal_future"]
        except KeyError:
            self.temporal = 0
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
    def __init__(self, data, last, betrayer, victim):
        self.year = int(data["season"])
        self.season = "spring" if (data["season"] - self.year) == 0 else "fall"
        self.interaction = data["interaction"]

        MessagePair = namedtuple("MessagePair", ["betrayer", "victim"])
        betrayer_messages = data["messages"]["betrayer"]
        victim_messages = data["messages"]["victim"]
        zipped = zip(betrayer_messages, victim_messages)
        self.messages = [MessagePair(betrayer=Message(m[0]), victim=Message(m[1])) for m in zipped]
        self.is_last_season_in_relationship = last
        self.betrayer = betrayer
        self.victim = victim

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

    def _sum_betrayer(self, attr):
        """
        Gets the sum of all messages for the given attribute in this Season.
        """
        return sum([getattr(m, attr) for m in self.get_messages('betrayer')])

    def _sum_victim(self, attr):
        """
        Gets the sum of all messages for the given attribute in this Season.
        """
        return sum([getattr(m, attr) for m in self.get_messages('victim')])

    def get_messages(self, person='betrayer'):
        """
        Generator for all the Message objects in this object.
        """
        for mp in self.messages:
            if person == "betrayer":
                yield mp.betrayer
            else:
                yield mp.victim

    def to_feature_vector(self, reverse=False, replicate=False):
        """
        Returns a feature vector version of this Season.

        If replicate is True, this feature vector will also include the number of temporal discourse tags.
        """
        b = [self._sum_betrayer('nwords'), self._sum_betrayer('nsentences'), self._sum_betrayer('nrequests'),
             self._avg_betrayer('politeness'), self._avg_betrayer('avg_sentiment')]
        v = [self._sum_victim('nwords'), self._sum_victim('nsentences'), self._sum_victim('nrequests'),
             self._avg_victim('politeness'), self._avg_victim('avg_sentiment')]
        if replicate:
            b.append(self._sum_betrayer('temporal'))
            v.append(self._sum_betrayer('temporal'))
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
        self.seasons = [Season(s, False, self.people[0], self.people[1]) for s in data["seasons"]]
        self.seasons[-1].is_last_season_in_relationship = True

    def __iter__(self):
        for s in self.seasons:
            yield s

    def __len__(self):
        return len(self.seasons)

    def __str__(self):
        s  = "ID: " + str(self.idx) + " "
        s += "Game: " + str(self.game) + " "
        s += "Betrayal: " + str(self.betrayal) + " "
        s += "People: " + str(self.people) + " "
        s += "Length of relationship: " + str(len(self.seasons)) + " "
        return s

    def get_next_season(self, season):
        """
        Gets the Season object that comes after the given one. If there isn't one, returns None.
        """
        for i, s in enumerate(self.seasons):
            if s == season:
                if i + 1 < len(self.seasons):
                    return self.seasons[i + 1]
                else:
                    return None
        return None

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


def get_all_sequences(datapath=None):
    """
    Gets all the relationship sequences and yields them one at a time except for the validation set.
    """
    global already_got_all_sequences
    if not already_got_all_sequences:
        dp = datapath if datapath else DATA_PATH
        with open(dp) as f:
            data = json.load(f)

        sequences = [Relationship(seq) for seq in data]
        random.shuffle(sequences)
        global validation_set
        global training_set
        validation_set = sequences[-50:]
        training_set = sequences[:-50]
        already_got_all_sequences = True
    for seq in training_set:
        yield seq

def _concatenate_trigram(tri, reverse, replicate=False):
    """
    Takes an iterable of three Season objects and returns a numpy vector of
    all the features of each Season concatenated.
    """
    fv = []
    for s in tri:
        fv += s.to_feature_vector(reverse, replicate)
    return np.array(fv)

def get_X_feed(reverse=True, datapath=None, upsample=False, replicate=False):
    """
    Generator for getting all the X vectors. Returns a tuple of (reversed, X) at each yield.

    Each X is a numpy array that looks like this:
    [N_words_Betrayer_season0, N_sentences_Betrayer_season0, N_requests_season0, politeness_season0, sentiment_season0, N_words_Victim_season0, ..., N_words_Victim_season1, ..., sentiment_season2]

    That is:
    [B, B, B, B, B, V, V, V, V, V          <- Season 0
     B, B, B, B, B, V, V, V, V, V          <- Season 1
     B, B, B, B, B, V, V, V, V, V]         <- Season 2

    When reverse is True, we will sometimes get:
    [V, V, V, V, V, B, B, B, B, B
     V, V, V, V, V, B, B, B, B, B
     V, V, V, V, V, B, B, B, B, B]

    When upsample is True, we add duplicate betrayal datapoints to address the class imbalance. These extra betrayals are all added to the end, so
    you should probably shuffle the data when you get it.

    The replicate parameter is whether or not you should include the planning discourse tags as well.
    """
    def get_them():
        for relationship in get_all_sequences(datapath):
            season_trigrams = relationship.get_season_trigrams()
            for tri in season_trigrams:
                r = random.choice([True, False]) if reverse else False
                yield relationship, tri, r, _concatenate_trigram(tri, r, replicate)

    if upsample:
        base = [(r, t) for _, _, r, t in get_them()]
        for i in range(UPSAMPLE_TIMES):
            base.extend([(r, t) for rel, tri, r, t in get_them() if rel.betrayal and tri[-1].is_last_season_in_relationship])
        for r, t in base:
            yield r, t
    else:
        for _, _, r, t in get_them():
            yield r, t

def get_validation_set(datapath=None, replicate=False):
    def get_Xs():
        for relationship in validation_set:
            season_trigrams = relationship.get_season_trigrams()
            for tri in season_trigrams:
                r = random.choice([True, False])
                yield _concatenate_trigram(tri, r, replicate)

    def get_ys():
        for i, relationship in enumerate(validation_set):
            season_trigrams = relationship.get_season_trigrams()
            for tri in season_trigrams:
                if tri[-1].is_last_season_in_relationship and relationship.betrayal:
                    yield 1
                else:
                    yield 0

    Y_val = np.array([y for y in get_ys()])
    X_val = np.array([x for x in get_Xs()])

    return X_val, Y_val

def get_X_feed_rnn(datapath=None):
    """
    Yields one relationship feature vector at a time: [[blah], [blah], [blah]] <-- One of these at a time; each one will be length 3 to 10
    """
    for relationship in get_all_sequences(datapath):
        yield [s.to_feature_vector(reverse=False) for s in relationship.seasons]

def get_Y_feed_binary_rnn(datapath=None):
    """
    Yields a list of 0s and 1s, where each slot in the yielded list is:
    1 if relationship is a betrayal AND this is the last season in the relationship. Otherwise 0.
    """
    for relationship in get_all_sequences(datapath):
        yield [1 if s.is_last_season_in_relationship and relationship.betrayal else 0 for s in relationship.seasons]

def _get_label_from_trigram(tri, relationship, betrayal, reverse):
    """
    Gets the Y corresponding to the given trigram.

    If reverse is True, then the returned label vector is:
    [Victim, Victim, Victim, B, B, B] rather than [B, B, B, V, V, V]

    The Y corresponding to a given trigram is whether B betrayed V or V betrayed B
    at one turn in the future, two, or three.

    IMPORTANT:
    When I say "one turn in the future", I mean the order phase that the messages come just before.
    """
    if not betrayal:
        # This relationship doesn't end in betrayal, so return all 0s
        return np.zeros(6)
    else:
        # Look at the last season in the trigram and the next two in the relationship
        # If there aren't two more in the relationship, any you are missing are 1s
        next_season = relationship.get_next_season(tri[-1])
        next_next_season = relationship.get_next_season(next_season)
        next_three = (tri[-1], next_season, next_next_season)
        if next_three[0].is_last_season_in_relationship:
            y = [1, 1, 1, 0, 0, 0]
        elif next_three[1].is_last_season_in_relationship:
            y = [0, 1, 1, 0, 0, 0]
        elif next_three[2].is_last_season_in_relationship:
            y = [0, 0, 1, 0, 0, 0]
        else:
            return np.zeros(6)

        if reverse:
            return np.array(y[3:] + y[:3], dtype=np.float32)
        else:
            return np.array(y, dtype=np.float32)


def get_Y_feed(X, datapath=None, upsample=False):
    """
    Generator for getting each Y vector (label vector) that corresponds to each X vector.
    X is a list of: [(reversed, fv), (reversed, fv), ...]

    A Y vector is a numpy array that looks like this:
    [A_betrays_B_in_one_season, A_betrays_B_in_two_seasons, A_betrays_B_in_three_seasons, B_betrays_A_in_one_season, B_betrays_A_in_two_seasons, B_betrays_A_in_three_seasons]

    IMPORTANT:
    As soon as someone betrays the other, the odds collapse to 1 and 0, so that if A betrays B in two seasons, then the vector will look like this:
    [0, 1, 1,
     0, 0, 0]

    When upsample is True, we add duplicate betrayal datapoints to address the class imbalance. These extra betrayals are all added to the end, so
    you should probably shuffle the data when you get it.
    UPSAMPLE IS NOT IMPLEMENTED FOR THIS FUNCTION YET.
    """
    for i, relationship in enumerate(get_all_sequences(datapath)):
        season_trigrams = relationship.get_season_trigrams()
        for tri in season_trigrams:
            yield _get_label_from_trigram(tri, relationship, relationship.betrayal, X[i][0])

def get_Y_feed_binary(datapath=None, upsample=False):
    """
    Generator for getting each y label (binary value) that corresponds to each X vector.

    The returned label indicates whether this triseason's last season is a betrayal (1) or not (0).

    When upsample is True, we add duplicate betrayal datapoints to address the class imbalance. These extra betrayals are all added to the end, so
    you should probably shuffle the data when you get it.
    """
    for i, relationship in enumerate(get_all_sequences(datapath)):
        season_trigrams = relationship.get_season_trigrams()
        for tri in season_trigrams:
            if tri[-1].is_last_season_in_relationship and relationship.betrayal:
                yield 1
            else:
                yield 0
    if upsample:
        for i in range(250 * UPSAMPLE_TIMES):
            yield 1

def x_str(x):
    """
    Takes a feature vector, x, and returns a pretty string representation of it.
    """
    s = "[ "
    s += "N words A Season 0: " + str(x[0]) + ", "
    s += "N sentences A Season 0: " + str(x[1]) + ", "
    s += "N requests A Season 0: " + str(x[2]) + ", "
    s += "Politeness A Season 0: " + str(x[3]) + ", "
    s += "Sentiment A Season 0: " + str(x[4]) + ", "

    s += "N words B Season 0: " + str(x[5]) + ", "
    s += "N sentences B Season 0: " + str(x[6]) + ", "
    s += "N requests B Season 0: " + str(x[7]) + ", "
    s += "Politeness B Season 0: " + str(x[8]) + ", "
    s += "Sentiment B Season 0: " + str(x[9]) + ", "

    s += "N words A Season 1: " + str(x[10]) + ", "
    s += "N sentences A Season 1: " + str(x[11]) + ", "
    s += "N requests A Season 1: " + str(x[12]) + ", "
    s += "Politeness A Season 1: " + str(x[13]) + ", "
    s += "Sentiment A Season 1: " + str(x[14]) + ", "

    s += "N words B Season 1: " + str(x[15]) + ", "
    s += "N sentences B Season 1: " + str(x[16]) + ", "
    s += "N requests B Season 1: " + str(x[17]) + ", "
    s += "Politeness B Season 1: " + str(x[18]) + ", "
    s += "Sentiment B Season 1: " + str(x[19]) + ", "

    s += "N words A Season 2: " + str(x[20]) + ", "
    s += "N sentences A Season 2: " + str(x[21]) + ", "
    s += "N requests A Season 2: " + str(x[22]) + ", "
    s += "Politeness A Season 2: " + str(x[23]) + ", "
    s += "Sentiment A Season 2: " + str(x[24]) + ", "

    s += "N words B Season 2: " + str(x[25]) + ", "
    s += "N sentences B Season 2: " + str(x[26]) + ", "
    s += "N requests B Season 2: " + str(x[27]) + ", "
    s += "Politeness B Season 2: " + str(x[28]) + ", "
    s += "Sentiment B Season 2: " + str(x[29])

    s += " ]"
    return s

def y_str(y):
    """
    Takes a label vector, y, and returns a pretty string representation of it.
    """
    s = "[ "
    s += "A betrays B this turn: " + str(y[0]) + ", "
    s += "A betrays B next turn: " + str(y[1]) + ", "
    s += "A betrays B turn after next: " + str(y[2]) + ", "
    s += "B betrays A this turn: " + str(y[3]) + ", "
    s += "B betrays A next turn: " + str(y[4]) + ", "
    s += "B betrays A turn after next: " + str(y[5])
    s += " ]"
    return s

if __name__ == "__main__":
    print("Getting data...")
    data = [d for d in get_all_sequences()]

    print("Length of data:", len(data))

    betrayals = [d for d in data if d.betrayal]
    print("Number of sequences that end in betrayal:", len(betrayals))

    Xs = [x for x in get_X_feed(reverse=False)]
    Ys = [y for y in get_Y_feed(Xs)]
    print("Length of Xs:", len(Xs))
    print("Length of Ys:", len(Ys))

