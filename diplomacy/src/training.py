"""
This is a front end module for running the program in training mode.

To use this module, simply either script it, or else load it in a Python3 shell session.
E.g.:
>>> import training
>>> training.train_tree("path/to/data", "path/to/save/model/at")
"""
import os
import data
import itertools
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib
if "SSH_CONNECTION" in os.environ:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
else:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        print("WARNING: This will work best if you install PyQt5")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import random
from sklearn import decomposition, neighbors, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier


np.set_printoptions(precision=2)
cached_Xs = None
cached_Ys = None
cached_ptd = None
cached_binary = True

def _get_xy(path_to_data=None, binary=True):
    """
    Returns Xs, Ys, shuffled.
    """
    global cached_Xs
    global cached_Ys
    global cached_ptd
    global cached_binary
    if cached_Xs is not None and cached_Ys is not None and cached_ptd == path_to_data and cached_binary == binary:
        return cached_Xs, cached_Ys
    else:
        print("Getting the data. This will take a moment...")
        upsample = True
        Xs = [x for x in data.get_X_feed(path_to_data, upsample=upsample)]
        if binary:
            Ys = np.array([y for y in data.get_Y_feed_binary(path_to_data, upsample=upsample)])
        else:
            Ys = np.array([y for y in data.get_Y_feed(Xs, path_to_data, upsample=upsample)])
        Xs = np.array([x[1] for x in Xs])

        # Shuffle
        index_shuf = [i for i in range(len(Xs))]
        random.shuffle(index_shuf)
        Xs = np.array([Xs[i] for i in index_shuf])
        Ys = np.array([Ys[i] for i in index_shuf])
        assert(len(Xs) == len(Ys))

        cached_Xs = Xs
        cached_Ys = Ys
        cached_ptd = path_to_data
        cached_binary = binary
        return Xs, Ys

def plot_confusion_matrix(cm, classes, subplot, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    """
    """
    plt.subplot(subplot)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
#    plt.colorbar()
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

def train_knn(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111, title=""):
    """
    Trains a knn classifier on the dataset.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    print("Training the KNN with inverse weights...")
    clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf = train_model(clf, cross_validate=True, conf_matrix=True, save_model_at_path=path_to_save_model, subplot=subplot, title=title)
    return clf

def train_logregr(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111, title=""):
    """
    Trains a logistic regression model.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    print("Training logistic regression model...")
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.1, fit_intercept=True,
                             intercept_scaling=1, class_weight='balanced', random_state=None, solver='liblinear', max_iter=200)
    clf = train_model(clf, cross_validate=True, conf_matrix=True, save_model_at_path=path_to_save_model, subplot=subplot, title=title)
    return clf

def train_mlp(path_to_data=None, path_to_save_model="mlp.hdf5", load_model=False, path_to_load="mlp.hdf5", binary=True, subplot=111, title=""):
    """
    Trains a multilayer perceptron.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    """
    print("Training the MLP...")
    def make_model():
        model = Sequential()
        model.add(Dense(128, input_dim=30, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        return model

    print("  |-> Getting the data...")
    Xs, Ys = _get_xy(path_to_data, binary)
    X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, random_state=0)

    if load_model:
        print("  |-> Loading saved model...")
        model = keras.models.load_model(path_to_load)
    else:
        model = make_model()

        print("  |-> Compiling...")
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        print("  |-> Fitting the model...")
        checkpointer = ModelCheckpoint(filepath=path_to_save_model, verbose=1, save_best_only=True)
        model.fit(X_train, y_train, batch_size=20, epochs=1000, verbose=2, validation_data=(X_test, y_test), callbacks=[checkpointer])

    print("  |-> Evaluating the model...")
    score = model.evaluate(X_test, y_test, verbose=1)
    print("")
    print("  |-> Loss:", score[0])
    print("  |-> Accuracy:", score[1])

    # Compute confusion matrix
    y_pred = model.predict(X_test)
    y_pred = [round(x[0]) for x in y_pred]
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=["No Betrayal", "Betrayal"], subplot=subplot, title=title)

def train_random_forest(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111, title=""):
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
    clf = train_model(clf, cross_validate=True, conf_matrix=True, save_model_at_path=path_to_save_model, subplot=subplot, title=title)
    return clf

def train_svm(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111, title=""):
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
    clf = train_model(clf, cross_validate=True, conf_matrix=True, save_model_at_path=path_to_save_model, subplot=subplot, title=title)
    return clf

def train_tree(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111, title=""):
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
    clf = train_model(clf, cross_validate=True, conf_matrix=True, save_model_at_path=path_to_save_model, subplot=subplot, title=title)
    return clf


def train_model(clf, cross_validate=False, conf_matrix=False, path_to_data=None, binary=True, save_model_at_path=None, subplot=111, title="Confusion Matrix"):
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
        plot_confusion_matrix(cnf_matrix, classes=["No Betrayal", "Betrayal"], subplot=subplot, title=title)

    if save_model_at_path:
        joblib.dump(clf, save_model_at_path)

    return clf

def load_model(path):
    """
    Returns a clf from the given path.
    """
    return joblib.load(path)

def pca_display(Xs, Ys):
    """
    """
    pca = decomposition.PCA(n_components=3)
    pca.fit(Xs)
    print("Here is how much variance is accounted for after reduction to 3D space:")
    print(pca.explained_variance_ratio_)
    X = pca.transform(Xs)

    fig = plt.figure(1, figsize=(4, 13))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

    plt.cla()
    y = Ys
    y = np.choose(y, [1, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)

    plt.show()


if __name__ == "__main__":
    Xs, Ys = _get_xy()
    ones = [y for y in Ys if y == 1]
    zeros = [y for y in Ys if y == 0]
    assert(len(ones) + len(zeros) == len(Ys))
    print("Betrayals:", len(ones))
    print("Non betrayals:", len(zeros))

    #pca_display(Xs, Ys)

    #train_mlp(load_model=True, path_to_load="mlps/mlp_269_240_29_598.hdf5", subplot=231, title="MLP")
    #train_knn(path_to_save_model="knn.model", subplot=232, title="KNN")
    #train_tree(path_to_save_model="tree.model", subplot=233, title="Tree")
    #train_random_forest(path_to_save_model="forest.model", subplot=234, title="Forest")
    #train_svm(path_to_save_model="svm.model", subplot=235, title="SVM")
    #train_logregr(path_to_save_model="logregr.model", subplot=236, title="Log Reg")

    train_mlp(load_model=True, path_to_load="mlps/mlp_269_240_29_598.hdf5", subplot=231, title="MLP")
    train_knn(load_model=True, path_to_load="models/knn.model", subplot=232, title="KNN")
    train_tree(load_model=True, path_to_load="models/tree.model", subplot=233, title="Tree")
    train_random_forest(load_model=True, path_to_load="models/forest.model", subplot=234, title="Forest")
    train_svm(load_model=True, path_to_load="models/svm.model", subplot=235, title="SVM")
    train_logregr(load_model=True, path_to_load="models/logregr.model", subplot=236, title="Log Reg")

    plt.show()

