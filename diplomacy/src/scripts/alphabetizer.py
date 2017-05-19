"""
This script fixes the randomness in the YAML files so that all a_to_b and b_to_a's are chosen
alphabetically.

You will need to run this script after the one that parses the game set.
"""
import itertools
import os
import sys

def rearrange(lines, old_name):
    # Get a_to_b from player
    # Get a_to_b to player
    # If a < b: we are good
    # else swap a_to_b's stuff with b_to_a's

    # Get player a and player b
    for i, line in enumerate(lines):
        if line.startswith("a_to_b:"):
            after_a_to_b = lines[i:]
            before_a_to_b = lines[:i]
            break
    for i, line in enumerate(after_a_to_b):
        if line.lstrip().startswith("from_player:"):
            _, player_a = line.split(":")
        if line.lstrip().startswith("to_player:"):
            _, player_b = line.split(":")
            break

    if player_a < player_b:
        # We are in alphabetical order
        return lines, old_name
    else:
        # We need to swap a_to_b and b_to_a
        a_to_b_line_generator = itertools.takewhile(lambda line: not line.startswith("b_to_a:"), after_a_to_b)
        b_to_a_line_generator = itertools.dropwhile(lambda line: not line.startswith("b_to_a:"), after_a_to_b)
        a_to_b_lines = [line for line in a_to_b_line_generator]
        b_to_a_lines = [line for line in b_to_a_line_generator]
        new_lines = before_a_to_b + b_to_a_lines + a_to_b_lines

        # Change the name
        old_letter_a, old_letter_b = old_name[-6], old_name[-5]
        new_letter_a, new_letter_b = old_letter_b, old_letter_a
        new_name = old_name[:-6] + new_letter_a + new_letter_b + ".yml"
        return new_lines, new_name

for path in os.listdir(sys.argv[1]):
    if path.endswith(".yml"):
        with open(os.path.join(sys.argv[1], path)) as f:
            lines = [line for line in f]
        lines, new_name = rearrange(lines, path)
        with open(os.path.join(sys.argv[1], new_name), 'w') as f:
            f.write(os.linesep.join(lines))
       os.remove(os.path.join(sys.argv[1], path))

