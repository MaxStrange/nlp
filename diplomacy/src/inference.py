"""
This is the API for the part of the program that does the inference.
"""
import os
if not "SSH_CONNECTION" in os.environ:
    # Disable annoying TF warnings when importing keras (which imports TF)
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import tensorflow as tf
import analyzer
import data
import keras
import numpy as np
from sklearn.externals import joblib


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
        return np.array(weighted_ys)



def get_relationship(rel_as_yam):
    """
    Creates a data.Relationship object from the given files.
    Used for inference, not training.
    """
    betrayal = False # Not needed for inference
    from_player = rel_as_yam[0]['a_to_b']['from_country']
    to_player = rel_as_yam[0]['a_to_b']['to_country']
    seasons = []
    for s in rel_as_yam:
        season = 0 if s['season'] == "Spring" else 0.5
        year = int(s['year']) + season
        interaction = None # Not needed for inference
        betrayer = "a_to_b" # Not needed for inference
        victim = "b_to_a" # Not needed for inference
        messages_betrayer = [analyzer.analyze_message(m) for m in s[betrayer]['messages']]
        messages_victim = [analyzer.analyze_message(m) for m in s[victim]['messages']]
        messages = {"betrayer": messages_betrayer, "victim": messages_victim}
        sdict = {"season": year, "interaction": interaction, "messages": messages}
        seasons.append(sdict)

    relationship = data.Relationship({"idx": 0, "game": 0, "betrayal": betrayal, "people": [from_player, to_player], "seasons": seasons})
    return relationship

def load_models():
    """
    Returns a list of classifiers loaded from the default model location.
    """
    model_paths = {
                    'KNN':      "models/knn.model",
                    'Tree':     "models/tree.model",
                    'Forest':   "models/forest.model",
                    'SVM':      "models/svm.model",
                    #'Logreg':   "models/logregr.model", <- This model ain't great
                  }
    models = [(name, joblib.load(path)) for name, path in model_paths.items()]
    models.append(("MLP", keras.models.load_model("models/mlp.hdf5")))
    models.append(("Ensemble", Ensemble([m[1] for m in models], [m[0] for m in models])))
    return models

def predict(rel):
    """
    Predicts whether there will be a betrayal or not next turn based on the given relationship.
    """
    assert len(rel) == 3, "Currently you need exactly 3 YAML files."
    fvs = []
    for s in rel.get_season_trigrams()[0]:
        fvs += s.to_feature_vector()
    Xs = np.array(fvs).reshape(1, -1)
    yes_nos = [(name, model.predict(Xs).tolist()[0]) for name, model in load_models()]
    yes_nos = [(name, round(yn[0])) if type(yn) == list else (name, yn) for name, yn in yes_nos]
    return yes_nos

