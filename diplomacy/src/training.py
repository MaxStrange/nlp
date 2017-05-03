"""
This is a front end module for running the program in training mode.

To use this module, simply either script it, or else load it in a Python3 shell session.
E.g.:
>>> import training
>>> training.train_svm("path/to/data", "path/to/save/model/at")
"""
import data

def _get_xy(path_to_data=None):
    """
    Returns Xs, Ys
    """
    Xs = data.get_X_feed(path_to_data)
    Ys = data.get_Y_feed(Xs, path_to_data)
    Xs = [x[1] for x in Xs]
    return Xs, Ys


def train_svm(path_to_data=None, path_to_save_model=None, load_model=False, path_to_load=None):
    """
    Trains an SVM classifier on the dataset.

    If no path_to_data is used, it will assume the default data directory.
    If no path_to_save_model is provided, it will save to the local directory.
    If load_model is True, it will load the model from the given location and resume training.
    """
    Xs, Ys = _get_xy(path_to_data)

def predict(models):
    """
    Uses each given model to predict and returns a dictionary of {model_name: prediction}
    """
    #TODO
    return None


if __name__ == "__main__":
    train_svm()
