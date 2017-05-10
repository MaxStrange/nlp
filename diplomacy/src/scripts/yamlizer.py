"""
This script takes a single message gathered from playdiplomacy.com as an input file and
outputs a YAML version of it so that it can be used as an input file into the program.

Usage:
python3 yamlizer.py msg.txt

(You can also feed it a list of files).

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

        get_content = lambda line: line.strip().split(":")[1].strip()
        self.from_ = get_content(from_line)
        self.date = get_content(date_line)
        self.to = get_content(to_line)
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
    print(message)
    exit()

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

