## This is the scripts directory

This directory is where I am keeping programs that run completely separately from betrayal.py.
See below for a list of each script in this directory:

- <b>yamlizer.py</b> This file may be of use to an end-user; it takes a message from playdiplomacy.com's format as an input
                     and returns a YAML version of it fit for betrayal.py.
- <b>gameparser.py</b> This file takes a directory and assumes it is a game; it takes all the .txt files in it (recursively)
                       and parses them into all combinations of players communicating with one another each turn. This is
                       too complicated to be of use to an end-user most likely.
