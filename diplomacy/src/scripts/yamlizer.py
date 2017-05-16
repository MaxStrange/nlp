"""
This script takes a single message gathered from playdiplomacy.com as an input file and
outputs a YAML version of it so that it can be used as an input file into the program.

Usage:
python3 yamlizer.py msg.txt

(You can also feed it a list of files).

NOTE: This script is not designed to handle multiple recipient messages. So if you want to
run this script on a message that has a CC line like:
CC: ENGLAND, FRANCE, ITALY
You must alter this to be:
CC: ENGLAND
or
CC: FRANCE
or
CC: ITALY
But there is nothing stopping you from running this script on the file multiple times - one for each recipient.

Example:

[Contents of msg.txt]:
=====================================================
From: ITALY
Date: May 07 2017 15:04 (GMT-8) Spring 1903 
CC: FRANCE 

Re:Movement 


Awesome.

About Bohemia... You should have Burgundy support Bohemia to Munich. Meanwhile, attack Ruhr with Belgium.

Unless you want me to contact Russia about what I should do with Bohemia. That would be fine too. Up to you.
======================================================

When fed through this script, this will output as msg.yml:

======================================================
year: 1903
season: Spring

a_to_b:
  from_player: Italy
  from_country: Italy
  to_player: France
  to_country: France
  messages:
    - >
      "Re:Movement 


      Awesome.

      About Bohemia... You should have Burgundy support Bohemia to Munich. Meanwhile, attack Ruhr with Belgium.

      Unless you want me to contact Russia about what I should do with Bohemia. That would be fine too. Up to you."
======================================================

You are free to do what you want with the from_player and to_player fields. Those are not really used - they were included because I didn't know if they would be
necessary or not - turns out not.

You can also add more messages to the end of that and also add b_to_a messages as well. See examples.
"""
import os
import sys

class Message:
    def __init__(self, txt):
        from_line = next((line for line in txt if line.lower().startswith("from:")))
        date_line = next((line for line in txt if line.lower().startswith("date:")))
        to_line = next((line for line in txt if line.lower().startswith("cc:")))
        to_line_index = [i for i, line in enumerate(txt) if line.lower().startswith("cc:")][0]
        msg_body = os.linesep.join([line for line in txt[(to_line_index + 1):]])

        self.from_ = from_line.split("From:")[1].strip()
        self.date = date_line.split("Date:")[1].strip()
        self.to = to_line.split("CC:")[1].strip()
        self.message = msg_body.strip()

    def __str__(self):
        s = "FROM: " + str(self.from_) + os.linesep
        s += "DATE: " + str(self.date) + os.linesep
        s += "TO: " + str(self.to) + os.linesep
        s += "Message: " + self.message
        return s

def yamlize(txt):
    """
    Converts the message text from playdiplomacy raw text to the YAML format that this program requires.
    """
    message = Message(txt)
    year = message.date.split(" ")[-1]
    season = message.date.split(" ")[-2]
    from_player = from_country = message.from_.lower().title()
    to_player = to_country = message.to.lower().title()
    msg = message.message.replace("\"", "\\\"")
    msg = "\"" + msg + "\""
    msg = msg.replace(os.linesep, os.linesep + "        ")

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
    if len(sys.argv) < 2:
        print("Need at least one file path.")
        exit()

    for file_path in sys.argv[1:]:
        try:
            with open(file_path) as f:
                text = [line for line in f]
            yaml = yamlize(text)
            with open(os.path.splitext(file_path)[0] + ".yml", 'w') as f:
                f.write(yaml)
        except FileNotFoundError:
            print("File", file_path, "does not exist.")

