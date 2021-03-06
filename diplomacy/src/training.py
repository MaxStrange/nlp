"""
This is a front end module for running the program in training mode.

This module was used to train the models and evaluate them.
"""
import os
if not "SSH_CONNECTION" in os.environ:
    # Disable annoying TF warnings when importing keras (which imports TF)
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import tensorflow as tf
import data
import itertools
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
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
import scipy
from sklearn import decomposition, neighbors, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

random.seed(12345)
np.random.seed(12345)
np.set_printoptions(precision=2)
cached_Xs = None
cached_Ys = None
cached_ptd = None
cached_binary = True
X_validation_set = None
Y_validation_set = None

def _get_rnn_data(path_to_data=None, binary=True):
    """
    """
    # Need to get whole relationships at a time along with labels that are simply whether this feature vector (which is a season)
    # is the last turn of a betrayal relationship

    # Dimension of Xs should be: (500, x, 10), where x varies from 3 to 10 (i.e., len of relationship)
    Xs = [x for x in data.get_X_feed_rnn(path_to_data)]
    if binary:
        Ys = [y for y in data.get_Y_feed_binary_rnn(path_to_data)]
    else:
        assert False, "Not yet supported"

    return Xs, Ys

def _get_xy(path_to_data=None, binary=True, upsample=True, replicate=False):
    """
    Returns Xs, Ys, shuffled.

    Keeps back a validation set that you can get via X_validation_set and Y_validation_set.
    """
    global cached_Xs
    global cached_Ys
    global cached_ptd
    global cached_binary
    if cached_Xs is not None and cached_Ys is not None and cached_ptd == path_to_data and cached_binary == binary and upsample and not replicate:
        return cached_Xs, cached_Ys
    else:
        print("Getting the data. This will take a moment...")
        Xs = [x for x in data.get_X_feed(path_to_data, upsample=upsample, replicate=replicate)]
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

        # Keep back validation set
        global X_validation_set
        global Y_validation_set
        X_validation_set, Y_validation_set = data.get_validation_set(replicate=replicate)
        print("Ones in validation set:", len([y for y in Y_validation_set if y == 1]))
        print("Zeros in validation set:", len([y for y in Y_validation_set if y == 0]))

        if upsample:
            # Only cache upsampled data
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
    plt.title(title, fontsize=20, fontweight='bold')
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=25,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    if subplot == 231 or subplot == 234:
        plt.ylabel('True label', fontsize=15)
    if subplot == 234 or subplot == 235 or subplot == 236:
        plt.xlabel('Predicted label', fontsize=15)

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
    if load_model:
        clf = load_model_from_path(path_to_load)
        compute_confusion_matrix(clf, upsample=False, subplot=subplot, title=title, path_to_data=path_to_data, binary=binary)
    else:
        clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
        clf = train_model(clf, cross_validate=True, conf_matrix=True, save_model_at_path=path_to_save_model, subplot=subplot, title=title)
    return clf

def train_logregr(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None, binary=True, subplot=111, title="", replicate=False):
    """
    Trains a logistic regression model.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    If binary is True, the model will be trained to simply detect whether, given three Seasons' worth of messages, there
        will be a betrayal between these users in this order phase.
    If replicate is True, this will attempt to train a model that corresponds to what the authors did.
    """
    print("Training logistic regression model...")
    if load_model:
        clf = load_model_from_path(path_to_load)
        compute_confusion_matrix(clf, upsample=False, subplot=subplot, title=title, path_to_data=path_to_data, binary=binary)
    else:
        clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.1, fit_intercept=True,
                             intercept_scaling=1, class_weight='balanced', random_state=None, solver='liblinear', max_iter=200)
        clf = train_model(clf, cross_validate=True, conf_matrix=True, save_model_at_path=path_to_save_model, subplot=subplot, title=title)
    return clf

def train_rnn(path_to_data=None, path_to_save_model="rnn.hdf5", load_model=False, path_to_load="rnn.hdf5", binary=True, subplot=111, title=""):
    """
    """
    def make_model(X):
        model = Sequential()
        model.add(LSTM(256, input_shape=X.shape[1:], dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    print("Training the RNN...")
    print("  |-> Getting the data...")
    X_train, y_train = _get_rnn_data(path_to_data, binary)
    X_test, y_test = X_validation_set, Y_validation_set

    def concat(Z):
        temp = []
        for z in Z:
            temp += z
        return temp

    print("  |-> Concatenating the data for the RNN...")
    X_train = np.array(concat(X_train))
    X_test = np.array(concat(X_test))
    y_train = np.array(concat(y_train))
    y_test = np.array(concat(y_test))

    print("  |-> X shape:", X_train.shape)
    print("  |-> Y shape:", y_train.shape)

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print("  |-> X shape after reshape:", X_train.shape)

    if load_model:
        print("  |-> Loading saved model...")
        model = keras.models.load_model(path_to_load)
    else:
        model = make_model(X_train)

        print("  |-> Compiling...")
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        print("  |-> Fitting the model...")
        checkpointer = ModelCheckpoint(filepath=path_to_save_model, verbose=1, save_best_only=True)
        model.fit(X_train, y_train, batch_size=10, epochs=1000, verbose=2, validation_data=(X_test, y_test), callbacks=[checkpointer])

    print("  |-> Evaluating the model...")
    score = model.evaluate(X_test, y_test, verbose=1)
    print("")
    print("  |-> Loss:", score[0])
    print("  |-> Accuracy:", score[1])

    compute_confusion_matrix(model, upsample=False, subplot=subplot, title=title, round_data=True)

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
        model.add(Dense(1024, input_dim=30, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        return model

    print("  |-> Getting the data...")
    X_train, y_train = _get_xy(path_to_data, binary)
    X_test = X_validation_set
    y_test = Y_validation_set

    if load_model:
        print("  |-> Loading saved model...")
        model = keras.models.load_model(path_to_load)
    else:
        model = make_model()

        print("  |-> Compiling...")
        model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])

        print("  |-> Fitting the model...")
        checkpointer = ModelCheckpoint(filepath=path_to_save_model, verbose=1, save_best_only=True)
        lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, min_lr=0.00001)
        model.fit(X_train, y_train, batch_size=20, epochs=1000, verbose=2, validation_data=(X_test, y_test), callbacks=[checkpointer, lr_reducer])

    print("  |-> Evaluating the model...")
    score = model.evaluate(X_test, y_test, verbose=1)
    print("")
    print("  |-> Loss:", score[0])
    print("  |-> Accuracy:", score[1])

    compute_confusion_matrix(model, upsample=False, subplot=subplot, title=title, round_data=True)
    return model

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
    if load_model:
        clf = load_model_from_path(path_to_load)
        compute_confusion_matrix(clf, upsample=False, subplot=subplot, title=title, path_to_data=path_to_data, binary=binary)
    else:
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
    if load_model:
        clf = load_model_from_path(path_to_load)
        compute_confusion_matrix(clf, upsample=False, subplot=subplot, title=title, path_to_data=path_to_data, binary=binary)
    else:
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
    if load_model:
        clf = load_model_from_path(path_to_load)
        compute_confusion_matrix(clf, upsample=False, subplot=subplot, title=title, path_to_data=path_to_data, binary=binary)
    else:
        clf = tree.DecisionTreeClassifier(class_weight='balanced')
        clf = train_model(clf, cross_validate=True, conf_matrix=True, save_model_at_path=path_to_save_model, subplot=subplot, title=title)
    return clf

def train_model(clf, cross_validate=False, conf_matrix=False, path_to_data=None, binary=True, save_model_at_path=None, subplot=111, title="Confusion Matrix", replicate=False):
    """
    Trains the given model.

    If confusion_matrix is True, a confusion matrix subplot will be added to plt.
    If path_to_data is specified, it will get the data from that location, otherwise it will get it from the default location.
    """
    X_train, y_train = _get_xy(path_to_data, binary, replicate=replicate)
    X_test, y_test = X_validation_set, Y_validation_set
    clf = clf.fit(X_train, y_train)
    if cross_validate:
        scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
        print("  |-> Scores:", scores)
    if confusion_matrix:
        compute_confusion_matrix(clf, upsample=False, subplot=subplot, title=title, path_to_data=path_to_data, binary=binary)
        #compute_roc_curve(clf, X_train, y_train, subplot=subplot, title=title)

    if save_model_at_path:
        joblib.dump(clf, save_model_at_path)

    return clf

def compute_roc_curve(clf, X, y, subplot=111, title="ROC"):
    """
    Computes and plots an ROC curve for the given classifier.
    """
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    classifier = clf
    classifier.probability=True

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = itertools.cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    i = 0
    for (train, test), color in zip(cv.split(X, y), colors):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += scipy.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.subplot(subplot)
        plt.plot(fpr, tpr, lw=lw, color=color, label='Fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

def compute_confusion_matrix(clf, upsample=True, subplot=111, title="Confusion Matrix", path_to_data=None, binary=True, round_data=False):
    """
    Computes and plots a confusion matrix.

    @param upsample is deprecated - instead, just change the upscale value in data.py
    """
    X_test, y_test = X_validation_set, Y_validation_set
    y_pred = clf.predict(X_test)
    if round_data:
        y_pred = [round(y[0]) for y in y_pred] # In case predicted value is from a model that does not output a binary value
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=["No Betrayal", "Betrayal"], subplot=subplot, title=title)
    print("Number of samples in validation set:", len(y_test))
    print("Number of betrayals in validation set:", sum(y_test))
    prfs = precision_recall_fscore_support(y_test, y_pred, average='micro')
    print("Precision, Recall, FScore, Support | Micro", prfs)
    prfs = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print("Precision, Recall, FScore, Support | Macro", prfs)
    prfs = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("Precision, Recall, FScore, Support | Weighted", prfs)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def load_model_from_path(path):
    """
    Returns a clf from the given path.
    """
    return joblib.load(path)

def pca_display(Xs, Ys, dimensions=3):
    """
    """
    assert dimensions == 2 or dimensions == 3, "Only 2D or 3D views are supported for pca_display"
    pca = decomposition.PCA(n_components=dimensions)
    print(Xs.shape)
    pca.fit(Xs)
    print("Here is how much variance is accounted for after dimension reduction:")
    print(pca.explained_variance_ratio_)
    X = pca.transform(Xs)

    fig = plt.figure(1, figsize=(4, 13))
    plt.clf()

    # Invert the Y array: [0 1 0 0 1 0] -> [1 0 1 1 0 1]
    print("Number of betrayals:", len([i for i in Ys if i == 1]))
    print("Number of non betrayals:", len([i for i in Ys if i == 0]))
    y = Ys
    y = np.choose(y, [1, 0]).astype(np.float)

    if dimensions == 3:
        ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

        plt.cla()
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.brg)
    else:
        plt.cla()
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg)

    plt.show()

class Ensemble:
    def __init__(self, models, names):
        self.models = models
        self.names = names

    def predict(self, Xs):
        ys = []
        for X in Xs:
            Y = []
            for name, model in zip(self.names, self.models):
                if name == "MLP":
                    Y.append(np.array([model.predict(X.reshape(1, 30)).tolist()[0]]))
                else:
                    Y.append(model.predict(X.reshape(1, -1)).tolist()[0])
            ys.append(Y)
        weighted_ys = []
        precision = 0.87 + 0.8+ 0.85 + 0.85 + 0.83
        recall = 0.89 + 0.78 + 0.63 + 0.71 + 0.69
        for Y in ys:
            weighted_Y = []
            for name, y in zip(self.names, Y):
                to_append = 0
                if name == "MLP" and y:
                    to_append = 1 * 0.89 / recall
                elif name == "MLP" and not y:
                    to_append = -1 * 0.87 / precision
                elif name == "KNN" and y:
                    to_append = 1 * 0.78 / recall
                elif name == "KNN" and not y:
                    to_append = -1 * 0.8 / precision
                elif name == "Tree" and y:
                    to_append = 1 * 0.63 / recall
                elif name == "Tree" and not y:
                    to_append = -1 * 0.85 / precision
                elif name == "Forest" and y:
                    to_append = 1 * 0.71 / recall
                elif name == "Forest" and not y:
                    to_append = -1 * 0.85 / precision
                elif name == "SVM" and y:
                    to_append = 1 * 0.69 / recall
                elif name == "SVM" and not y:
                    to_append = -1 * 0.83 /precision
                else:
                    assert False, "Model: " + name + " not accounted for when y is " + bool(y)
                weighted_Y.append(to_append)
            prediction = np.array([1]) if sum(weighted_Y) > 0.51 else np.array([0])
            weighted_ys.append(prediction)
        return weighted_ys

if __name__ == "__main__":
    Xs, Ys = _get_xy()
    ones = [y for y in Ys if y == 1]
    zeros = [y for y in Ys if y == 0]
    assert(len(ones) + len(zeros) == len(Ys))
    print("Betrayals:", len(ones))
    print("Non betrayals:", len(zeros))

    pca_display(Xs, Ys, dimensions=2)

    #train_rnn(path_to_save_model="rnn.hdf5", subplot=236, title="RNN")
    #mlp = train_mlp(path_to_save_model="mlp.hdf5", subplot=231, title="MLP")
    #knn = train_knn(path_to_save_model="knn.model", subplot=232, title="KNN")
    #tree =train_tree(path_to_save_model="tree.model", subplot=233, title="Tree")
    #forest = train_random_forest(path_to_save_model="forest.model", subplot=234, title="Forest")
    #svm = train_svm(path_to_save_model="svm.model", subplot=235, title="SVM")
    #train_logregr(path_to_save_model="logregr.model", subplot=236, title="Log Reg")

    mlp = train_mlp(load_model=True, path_to_load="models/mlp.hdf5", subplot=231, title="MLP")
    knn = train_knn(load_model=True, path_to_load="models/knn.model", subplot=232, title="KNN")
    tree = train_tree(load_model=True, path_to_load="models/tree.model", subplot=233, title="Tree")
    forest = train_random_forest(load_model=True, path_to_load="models/forest.model", subplot=234, title="Forest")
    svm = train_svm(load_model=True, path_to_load="models/svm.model", subplot=235, title="SVM")
    #rnn = train_rnn(load_model=True, path_to_load="models/rnn.hdf5", subplot=236, title="RNN")

    #logregr = train_logregr(load_model=True, path_to_load="models/logregr.model", subplot=236, title="Log Reg", replicate=True)

    ensemble = Ensemble([mlp, knn, tree, forest, svm], ["MLP", "KNN", "Tree", "Forest", "SVM"])
    print("Computing the ensemble...")
    compute_confusion_matrix(ensemble, upsample=False, subplot=236, title="Ensemble")
    plt.show()

