"""
This module provides a command line interface for making graphs and charts of the data.
"""
import argparse
import betrayal
import matplotlib.pyplot as plt
import os
import sys

def plot_triplet(relationship):
    """
    Plots the given triplet/relationship.
    """
    fvs = [s.to_feature_vector() for s in relationship]
    b_nwords = [fv[0] for fv in fvs]
    b_nsents = [fv[1] for fv in fvs]
    b_nrequs = [fv[2] for fv in fvs]
    b_polite = [fv[3] for fv in fvs]
    b_sentim = [fv[4] for fv in fvs]
    v_nwords = [fv[5] for fv in fvs]
    v_nsents = [fv[6] for fv in fvs]
    v_nrequs = [fv[7] for fv in fvs]
    v_polite = [fv[8] for fv in fvs]
    v_sentim = [fv[9] for fv in fvs]
    b = relationship.people[0]
    v = relationship.people[1]
    # TODO: Plot each of these lists over time, b and v on the same plot
    print(b, v) # France, Russia

    def plot_feature(subplotnum, title, xlabel, line, color, plot_label):
        plt.subplot(subplotnum)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.plot(line, color, label=plot_label)
        plt.legend()
        plt.tight_layout()

    plot_feature(231, "Number of words", "Turn", b_nwords, "ro-", b)
    plot_feature(231, "Number of words", "Turn", v_nwords, "b^--", v)

    plot_feature(232, "Number of Sentences", "Turn", b_nsents, "ro-", b)
    plot_feature(232, "Number of Sentences", "Turn", v_nsents, "b^--", v)

    plot_feature(233, "Number of Requests", "Turn", b_nrequs, "ro-", b)
    plot_feature(233, "Number of Requests", "Turn", v_nrequs, "b^--", v)

    plot_feature(234, "Avg Message Politeness", "Turn", b_polite, "ro-", b)
    plot_feature(234, "Avg Message Politeness", "Turn", v_polite, "b^--", v)

    plot_feature(235, "Avg Message Sentiment", "Turn", b_sentim, "ro-", b)
    plot_feature(235, "Avg Message Sentiment", "Turn", v_sentim, "b^--", v)

    plt.show()

def execute(args):
    """
    Execute the program based on the given arguments.
    Returns whether the program did something or not.
    """
    if args.triplet:
        betrayals, relationship = betrayal.betrayal(args.triplet)
        print("Betrayals:", betrayals)
        print("Relationship:", relationship)
        plot_triplet(relationship)
        return True

    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is the visualization CLI for the diplomacy dataset.")
    parser.add_argument("-t", "--triplet", help="Give the CLI three YAML files to make a relationship out of and have it graph the resulting realtionship with the charting suite.",
                        nargs=3, metavar=("path/to/file1.yml", "path/to/file2.yml", "path/to/file3.yml"))

    args = parser.parse_args()
    did_something = execute(args)
    if not did_something:
        print("----------------------------------------------")
        print("ERROR: You need to supply at least one argument.")
        print("----------------------------------------------")
        parser.print_help()
        exit()
