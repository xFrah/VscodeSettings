# open file and print it without newlines

import sys


def remove_newlines(filename):
    with open(filename) as f:
        for line in f:
            print(line.rstrip(), end="")


remove_newlines("ricevuta_commands.json")
