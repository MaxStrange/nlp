"""
This is a front end module for running the program in training mode.

To use this module, simply either script it, or else load it in a Python3 shell session.
E.g.:
>>> import training
>>> training.train_tree("path/to/data", "path/to/save/model/at")
"""
import data
import itertools
import matplotlib.pyplot as plt
try:
    plt.switch_backend("Qt5Agg")
except Exception:
    print("WARNING: This will work best if you install PyQt5")
import numpy as np
from sklearn import neighbors, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier


np.set_printoptions(precision=2)

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

def plot_confusion_matrix(cm, classes, subplot, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    """
    """
    plt.subplot(subplot)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def train_knn(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111):
    """
    Trains a knn classifier on the dataset.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    print("Training the KNN with uniform weights...")
    clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform')
    clf = train_model(clf, cross_validate=True, conf_matrix=True, subplot=subplot)

def train_mlp(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111):
    """
    Trains a multilayer perceptron.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    print("Training the MLP...")
    clf = MLPClassifier(solver='sgd', alpha=1e-5, learning_rate='invscaling', max_iter=2000000, shuffle=True, tol=1e-10, verbose=True, hidden_layer_sizes=(25, 2), random_state=1)
    clf = train_model(clf, cross_validate=True, conf_matrix=True, subplot=subplot)

def train_random_forest(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111):
    """
    Trains a random forest classifier on the dataset.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    print("Training the random forest...")
    clf = RandomForestClassifier(class_weight='balanced')
    clf = train_model(clf, cross_validate=True, conf_matrix=True, subplot=subplot)

def train_svm(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111):
    """
    Trains an SVM classifier on the dataset.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    print("Training the SVM with nonlinear kernel (RBF)...")
    clf = svm.SVC(class_weight='balanced')
    clf = train_model(clf, cross_validate=True, conf_matrix=True, subplot=subplot)

def train_tree(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111):
    """
    Trains a decision tree classifier on the dataset.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    print("Training the decision tree model...")
    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    clf = train_model(clf, cross_validate=True, conf_matrix=True, subplot=subplot)


def train_model(clf, cross_validate=False, conf_matrix=False, path_to_data=None, binary=True, save=False, model_path="model.pkl", subplot=111):
    """
    Trains the given model.

    If confusion_matrix is True, a confusion matrix subplot will be added to plt.
    If path_to_data is specified, it will get the data from that location, otherwise it will get it from the default location.
    """
    Xs, Ys = _get_xy(path_to_data, binary)
    X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, random_state=0)
    clf = clf.fit(X_train, y_train)
    if cross_validate:
        scores = cross_val_score(clf, Xs, Ys, cv=5, n_jobs=-1)
        print("  |-> Scores:", scores)
    if confusion_matrix:
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cnf_matrix, classes=["No Betrayal", "Betrayal"], subplot=subplot)

    if save:
        joblib.dump(clf, model_path)

    return clf

if __name__ == "__main__":
    Xs, Ys = _get_xy()
    ones = [y for y in Ys if y == 1]
    zeros = [y for y in Ys if y == 0]
    assert(len(ones) + len(zeros) == len(Ys))
    print("Betrayals:", len(ones))
    print("Non betrayals:", len(zeros))
    #train_mlp()
    #train_knn()
    train_tree(subplot=131)
    train_random_forest(subplot=132)
    train_svm(subplot=133)
    plt.show()

