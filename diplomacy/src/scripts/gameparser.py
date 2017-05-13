"""
Takes a directory as input and walks through it recursively looking for
all .txt files. Each file should contain complete messages in the format of
playdiplomacy.com and separated by some number of "=========="'s on a new line.

It also takes a file that is just the playdiplomacy gamehistory page saved.

The output of this script is a .yml file for each combination of players who
communicated with one another per season over the course of the whole game.
"""
from datetime import datetime
import itertools
import os
import re
import sys

class Time:
    def __init__(self, line):
        self.season, self.year, self.phase, _,  _month, _day, _year, self.time, _tz = line.strip().split(" ")
        self.time = datetime.strptime(self.time, "%H:%M")

    def __eq__(self, other):
        for name, value in self.__dict__.items():
            print(name, ":", value)

            if hasattr(other, name):
                if value != getattr(other, name):
                    return False
            else:
                return False
        return True

class TimeList:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.times = [Time(line) for line in f if line.strip() and line.strip() != "Winter 1900" and line.strip() != "Fall 1905 Retreat"]

    def after_deadline(self, season, year, time, msg):
        """
        Takes:
        Fall/Spring, 19xx, datetime, msg

        and returns whether the datetime is after the deadline for that phase.

        Example:
        Fall 1904 16:53 -> True
        The reason this returns True is because the Fall 1904 Orders phase ended at 16:45; we are actually into the build phase.
        """
        for t in self.times:
            if t.season == season and t.year == year and t.phase == "Orders":
                if time < t.time:
                    return False
                elif time == t.time:
                    # Message was sent in the minute that the phase ended. If the message is very short, this may have been when it was written.
                    # Otherwise, the message was probably written and then sent right as the phase ended, and really belongs in the phase.
                    if len(msg) < 50:
                        return True
                    else:
                        return False
                else:
                    return True

        def __eq__(self, other):
            for name, value in self.__dict__.items():
                print(name, ":", value)

                if hasattr(other, name):
                    if value != getattr(other, name):
                        return False
                else:
                    return False
            return True

class Message:
    def __init__(self, txt, timelist):
        txt = txt.split(os.linesep)
        from_line = next((line for line in txt if line.lower().startswith("from:")))
        date_line = next((line for line in txt if line.lower().startswith("date:")))
        to_line = next((line for line in txt if line.lower().startswith("cc:")))
        to_line_index = [i for i, line in enumerate(txt) if line.lower().startswith("cc:")][0]
        msg_body = os.linesep.join([line for line in txt[(to_line_index + 1):]])

        self.from_ = from_line.split("From:")[1].strip().lower().title()

        date = date_line.split("Date:")[1].strip()
        self.year = int(date.split(" ")[-1])
        self.season = date.split(" ")[-2]

        self.to = to_line.split("CC:")[1].strip().lower().title()
        self.message = msg_body.strip()
        self.time = datetime.strptime(date_line.split(" ")[4], "%H:%M")

        # Change season if after deadline
        if timelist.after_deadline(self.season, self.year, self.time, self.message):
            self.year = self.year + 1 if self.season == "Fall" else self.year
            self.season = "Fall" if self.season == "Spring" else "Spring"

    def __eq__(self, other):
        for name, value in self.__dict__.items():
            print(name, ":", value)

            if hasattr(other, name):
                if value != getattr(other, name):
                    return False
            else:
                return False
        return True

    def __hash__(self):
        return hash(frozenset(self.__dict__.items()))

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = "FROM: " + str(self.from_) + os.linesep
        s += "YEAR: " + str(self.year) + os.linesep
        s += "SEASON: " + str(self.season) + os.linesep
        s += "TO: " + str(self.to) + os.linesep
        s += "TIME: " + self.time.strftime("%H:%M") + os.linesep
        s += "Message: " + self.message
        return s


def yamlize(txt):
    """
    Converts the message text from playdiplomacy raw text to the YAML format that this program requires.
    """
    message = Message(txt)
    msg = message.message.replace("\"", "\\\"")
    msg = "\"" + msg + "\""

    yaml = "year: " + year + os.linesep
    yaml += "season: " + season + os.linesep
    yaml += "a_to_b:" + os.linesep
    yaml += "  from_player: " + from_player + os.linesep
    yaml += "  from_country: " + from_country + os.linesep
    yaml += "  to_player: " + to_player + os.linesep
    yaml += "  to_country: " + to_country + os.linesep
    yaml += "  messages:" + os.linesep
    yaml += "    - >" + os.linesep
    yaml += msg
    yaml += os.linesep

    return yaml

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Need a directory and a gamehistory file.")
        exit()
    else:
        path = sys.argv[1]
        timepath = sys.argv[2]

    filepaths = []
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            filepaths.append(os.path.join(dirpath, name))
    filepaths = [path for path in filepaths if path.endswith(".txt")]

    texts = []
    for path in filepaths:
        with open(path, 'r') as f:
            s = ""
            for line in f:
                if line.startswith("============="):
                    texts.append(s)
                    s = ""
                    continue
                else:
                    s += line + os.linesep

    timelist = TimeList(timepath)
    msgs = [Message(t, timelist) for t in texts]

    # Lump into conversations
    pair_maybe_conversation = lambda a, b: a.year == b.year and a.season == b.season
    combos = (pair for pair in itertools.combinations(msgs, 2) if pair_maybe_conversation(*pair))
    pair_is_to_each_other = lambda a, b: set((a.to, a.from_)) == set((b.to, b.from_))
    conversation_pairs = [(a, b) for a, b in combos if pair_is_to_each_other(a, b)]
    # At this point, you have pairs of messages that belong together. But now you need to group all pairs that are the same into lists
    def make_key(cp):
        letters = sorted((cp[0].to[0], cp[0].from_[0]))
        return str(cp[0].year) + cp[0].season + letters[0] + letters[1] # --> E.g., 1901SpringAF

    def countries_are_same(cp, other):
        return set([cp[0].to, cp[0].from_]) == set([other[0].to, other[0].from_])

    conversations = {}
    for cp in conversation_pairs:
        for other in conversation_pairs:
            if cp is not other and cp[0].year == other[0].year and cp[0].season == other[0].season and countries_are_same(cp, other):
                key = make_key(cp)
                to_add = [cp[0], cp[1], other[0], other[1]]
                try:
                    conversations[key] += to_add
                except KeyError:
                    conversations[key] = to_add

    conversations = [list(set([msg for msg in v])) for _k, v in conversations.items()]

