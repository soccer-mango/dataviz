# TODO: All the code to produce your graphs for Stage 1 goes here!
from tracemalloc import stop
from utils import get_ri_stops_df, get_banknote_df
import matplotlib.pyplot as plt
import pandas as pd

def graph1():

    # Get data
    stops_data = get_ri_stops_df()

    stops_data['stop_outcome'].value_counts().plot.bar()
    plt.title("Composition of Stop Outcomes")
    plt.xlabel("Stop Outcomes")
    plt.ylabel("Totals")
    plt.show()

def graph2():

    #Get data
    bank_data = get_banknote_df()

    df = pd.DataFrame()
    df["z"] = bank_data.loc[:,"Skewness"]
    df["y"] = bank_data.loc[:,"Variance"]
    df["x"] = bank_data.loc[:,"Class"]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(df["x"], df["y"], df["z"])

    plt.title("Variance and Skewness of Forged Bank Note (1) vs non Forges Note (0)")
    plt.xlabel("Class: (1) for Forged, (0) for Clean")
    plt.ylabel("Variance")

    plt.show()

def graph3():

    #Get data
    stops_data = get_ri_stops_df()

    fig, ax = plt.subplots()

    stops_data['driver_race'].value_counts().plot(kind="pie", label="Race", ax=ax, y="Race")
    plt.title("Racial Composition of Police Stops")
    plt.show()

graph1()
graph2()
graph3()