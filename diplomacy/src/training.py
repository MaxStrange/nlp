"""
This is a front end module for running the program in training mode.

To use this module, simply either script it, or else load it in a Python3 shell session.
E.g.:
>>> import training
>>> training.train_tree("path/to/data", "path/to/save/model/at")
"""
import data
import numpy as np
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def _get_xy(path_to_data=None, binary=True):
    """
    Returns Xs, Ys
    """
    Xs = [x for x in data.get_X_feed(path_to_data)]
    if binary:
        Ys = np.array([y for y in data.get_Y_feed_binary(path_to_data)])
    else:
        Ys = np.array([y for y in data.get_Y_feed(Xs, path_to_data)])
    Xs = np.array([x[1] for x in Xs])
    return Xs, Ys


def train_random_forest(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True):
    """
    Trains a random forest classifier on the dataset.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    Xs, Ys = _get_xy(path_to_data, binary)
    clf = RandomForestClassifier(class_weight='balanced')
    print("Training the random forest...")
    scores = cross_val_score(clf, Xs, Ys, cv=5)
    print("  |-> Scores:", scores)

def train_svm(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True):
    """
    Trains an SVM classifier on the dataset.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    Xs, Ys = _get_xy(path_to_data, binary)
    clf = svm.SVC(class_weight='balanced')
    print("Training the SVM...")
    scores = cross_val_score(clf, Xs, Ys, cv=5)
    print("  |-> Scores:", scores)


def train_tree(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True):
    """
    Trains a decision tree classifier on the dataset.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    Xs, Ys = _get_xy(path_to_data, binary)
    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    print("Training the decision tree model...")
    scores = cross_val_score(clf, Xs, Ys, cv=5)
    print("  |-> Scores:", scores)

def predict(models):
    """
    Uses each given model to predict and returns a dictionary of {model_name: prediction}
    """
    #TODO
    return None


if __name__ == "__main__":
    Xs, Ys = _get_xy()
    ones = [y for y in Ys if y == 1]
    zeros = [y for y in Ys if y == 0]
    assert(len(ones) + len(zeros) == len(Ys))
    print("Ones:", len(ones))
    print("Zeros:", len(zeros))
    train_tree()
    train_random_forest()
    train_svm()

