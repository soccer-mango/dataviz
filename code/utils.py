import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn import svm

from sklearn import tree

############### PANDAS DATA IMPORT HELPER FUNCTIONS #########################

def get_ri_stops_df():
    """
    Helper function. Will return a Pandas DataFrame for the RI Traffic Stops dataset
    """
    ri_stops_path = getcurdir() + f"{os.sep}..{os.sep}data{os.sep}ri_traffic_stops.csv"
    return pd.read_csv(ri_stops_path)


def get_banknote_df():
    """
    Helper function. Will return a Pandas DataFrame for the Banknote Authentication Dataset
    """
    banknote_path = getcurdir() + f"{os.sep}..{os.sep}data{os.sep}banknote_authentication.csv"
    return pd.read_csv(banknote_path)
    
    
############### MACHINE LEARNING MODELS ###############

TEST_SIZE = 0.2 # TODO: Feel free to modify this!
KNN_NUM_NEIGHBORS = 5 # TODO: Feel free to modify this!
RANDOM_SEED = 0


def get_trained_model(dataset_name, model_name, target_name=None, feature_names=None, k=KNN_NUM_NEIGHBORS):
    """
    Input:
    - dataset_name: str, a dataset name. One of ["ri_traffic_stops", "banknote_authentication"]
    - A model name: str, a model name. One of ["decision_tree", "k_nearest_neighbor", "logistic_regression", "dummy"]
    - target_name: str, the name of the variable that you want to make the label of the regression/classification model
    - feature_names: list[str], the list of strings of the variable names that you want to be the features of your
                        regression/classification model
    
    What it does:
    - Create a model that matches with the model_name, and then fit to the One-Hot-Encoded dataset

    Output: A tuple of four things:
    - model: The model associated with the model_name - **TRAINED**
    - ohe: The one hot encoder that is used to encode non-numeric features in the dataset
    - train_df: The training dataset (DataFrame)
    - test_df: The testing dataset (DataFrame)
    """
    # first, check if the inputs are valid
    assert dataset_name in ["ri_traffic_stops", "banknote_authentication"], \
            f"Invalid input for function get_model: dataset_name = {dataset_name}, supposed to be in ['ri_traffic_stops', 'banknote_authentication']"
    assert model_name in ["decision_tree", "k_nearest_neighbor", "logistic_regression", "dummy"], \
            f"Invalid input for function get_model: model_name = {model_name}, supposed to be in ['decision_tree', 'k_nearest_neighbor', 'logistic_regression', 'dummy']"
    
    # creating a OneHotEncoder for non-numeric features
    ohe = OneHotEncoder(handle_unknown='ignore')

    # getting the exact model - the formatting is a lil cursed :P
    if model_name == "decision_tree": model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    if model_name == "logistic_regression": model = LogisticRegression(random_state=RANDOM_SEED)
    if model_name == "k_nearest_neighbor": model = KNeighborsClassifier(n_neighbors=KNN_NUM_NEIGHBORS)
    # this model is a dummy model - for baseline model! :)
    if model_name == "dummy": model = DummyClassifier(random_state=RANDOM_SEED)

    if dataset_name == "ri_traffic_stops":
        """
        Default assumption: target label is `stop_outcome`, feature_names are the rest
        """
        data = get_ri_stops_df()
        if target_name == None:
            target_name = "stop_outcome" # default assumption

        if feature_names == None:
            feature_names = [e for e in data.columns if e != target_name] # default assumption

    if dataset_name == "banknote_authentication":
        """
        Assumption: target label is `Class`, feature_names are the rest
        """
        data = get_banknote_df()
        if target_name == None:
            target_name = "Class" # default assumption

        if feature_names == None:
            feature_names = [e for e in data.columns if e != target_name] # default assumption

    # now assert a few things to make sure all the column names are valid
    assert target_name in data.columns, f"Column not found: {target_name}"
    for lbl in feature_names:
        assert lbl in data.columns, f"Column not found: {lbl}"
    

    train_df, test_df = train_test_split(data, test_size=TEST_SIZE)

    model.fit(ohe.fit_transform(train_df[feature_names]), train_df[target_name])
    return model, ohe, train_df, test_df
    

def get_model_accuracy(model, df, one_hot_encoder, dataset_name=None, target_name=None, feature_names=None):
    """
    Inputs:
    - model: sklearn model that was returned by get_model (or created yourself)
    - df: The dataframe that contains the features and the target
    - one_hot_encoder: The sklearn OneHotEncoder that was returned by get_model (or created yourself)
                        that learns how to encode the non-numeric features in the dataset df
    - dataset_name: if not None, has to be one of ["ri_traffic_stops", "banknote_authentication"]
    - target_name, feature_names: if not None, has to be in df.columns

    Outputs: A tuple of three things:
    - acc: Accuracy score
    - y_pred: The model's predictions (numpy array)
    - y_targ: The target labels (numpy array)
    """
    ##### INPUT ASSERTIONS #####
    # if either target_name == None or feature_names == None, dataset_name has to be != None
    if target_name == None or feature_names == None:
        assert dataset_name in ["ri_traffic_stops", "banknote_authentication"], \
            """if either target_name == None or feature_names == None, dataset_name has to be != None.
                Check input to get_model_accuracy"""
    
    # if nothing is passed to target_name and feature_names, we'll use the default
    default_targ_label = {
        "ri_traffic_stops": "stop_outcome",
        "banknote_authentication": "Class"
    }

    # if nothing was inputted into target_lalbel, use the default target label
    if target_name == None:
        target_name = default_targ_label[dataset_name]
    
    if feature_names == None:
        feature_names = [e for e in df.columns if e != target_name]

    # and then assert that target_name and each of feature_names in df
    assert target_name in df.columns, f"Column not found: {target_name}"
    for lbl in feature_names:
        assert lbl in df.columns, f"Column not found: {lbl}"


    ##### Alright. Now onto the meat of the function! #####
    # encode the features
    encoded = one_hot_encoder.transform(df[feature_names])

    # and then use the model to predict
    y_pred = model.predict(encoded)
    
    # get the y_target
    y_targ = df[target_name].to_numpy()

    # get the accuracy
    acc = (y_pred == y_targ).sum() / len(y_pred)    
    
    return acc, y_pred, y_targ


############### UTILS HELPER FUNCTIONS ###############

def getcurdir():
    """
    Helper function to get the absolute path of utils.py
    """
    return os.path.dirname(os.path.realpath(__file__))

    
    