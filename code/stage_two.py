# TODO: All the code to produce your graphs for Stage 2 goes here!
from utils import get_ri_stops_df, get_trained_model, get_model_accuracy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def model_generator(dataset_name, model_name, target_name, feature_names):

    trained_model, one_hot_encoder, train_df, test_df = get_trained_model(dataset_name=dataset_name,\
                                                                        model_name=model_name,\
                                                                        target_name=target_name,\
                                                                        feature_names=feature_names)
    
    training_acc, training_preds, training_targs = get_model_accuracy(model=trained_model,\
                                                                        df=train_df,\
                                                                        one_hot_encoder=one_hot_encoder,\
                                                                        dataset_name=dataset_name,\
                                                                        target_name=target_name,\
                                                                        feature_names=feature_names)

    testing_acc, testing_preds, testing_targs = get_model_accuracy(model=trained_model,\
                                                                    df=test_df,\
                                                                    one_hot_encoder=one_hot_encoder,\
                                                                    dataset_name=dataset_name,\
                                                                    target_name=target_name,\
                                                                    feature_names=feature_names)

    return trained_model

model1 = model_generator("ri_traffic_stops", "decision_tree", "stop_outcome", ["driver_race", "driver_gender"])

def plot_model1():

    tree.plot_tree(model1)
    plt.show()

plot_model1()

def plot_model2():
    ks = []
    accuracies = []

    for k in range(3, 8):
        print("hi")
        ks.append(k)
        trained_model, one_hot_encoder, train_df, test_df = get_trained_model(dataset_name="ri_traffic_stops",\
                                                                        model_name="k_nearest_neighbor",\
                                                                        target_name="stop_time",\
                                                                        feature_names=["driver_age"], k=k)
        testing_acc, training_preds, training_targs = get_model_accuracy(model=trained_model,\
                                                                        df=test_df,\
                                                                        one_hot_encoder=one_hot_encoder,\
                                                                        dataset_name="ri_traffic_stops",\
                                                                        target_name="stop_time",\
                                                                        feature_names=["driver_age"])
        accuracies.append(testing_acc)
    
    plt.bar(ks, accuracies)
    plt.show()

plot_model2()

def plot_model3():

    data = get_ri_stops_df()
    data["search_conducted"].replace({"FALSE": 0, "TRUE": 1})
    
    featured = data[["driver_age", "search_conducted"]]

    logreg = LogisticRegression()
    logreg.fit(featured, featured["search_conducted"])

    prediction = logreg.predict(featured)

    cf = confusion_matrix(data["search_conducted"], prediction)
    sns.heatmap(cf, annot=True)
    plt.show()

plot_model3()