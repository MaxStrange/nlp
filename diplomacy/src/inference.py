"""
This is the API for the part of the program that does the inference.
"""
import analyzer
import data
import keras
import numpy as np
from sklearn.externals import joblib


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

