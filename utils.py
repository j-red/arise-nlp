from tsfresh import extract_features, extract_relevant_features
import sys, os, math, random, glob, torch, pickle, json
from pprint import pprint # pretty print, useful for results dictionaries
import numpy as np
import pandas as pd
from sklearn import *

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt


ROOT_DIR = os.path.dirname(os.path.abspath("."))
def get_full_path(*path):
    """ Returns the full path to a specific directory. From utils.py. """
    return os.path.join(ROOT_DIR, *path)


DATA_DIR = get_full_path("data_28") + "/"
FIGURE_DIR = get_full_path("MTL", "Figures") + "/"


def get_i_datasets(numSets):
    ''' Returns an aggregated DataFrame object containing data from the first i datasets in ../data_28/ '''
    master = pd.DataFrame(columns=['id', 'index', 'datetime', 'rtt']) # Create empty dataframe to append new DataFrames to.

    for i in range(numSets):
        f = DATA_DIR + f"dataset_{i:02d}.csv" # get file path
        df = pd.read_csv(f) # Read raw csv
        df.rename(columns = {'Unnamed: 0':'index'}, inplace = True) # rename index column to 'index'

        df.insert(0, 'id', i) # create and populate ID column as first entry in dataframe
        master = master.append(df, ignore_index = True)
    master['rtt'] = master['rtt'].round(3) # round to 3 decimals places
    return master

def get_all_datasets(count=28):
    """ Returns a DataFrame containing all of the initial datasets. """
    # count is the number of dataset CSVs contained in ROOT/data_28/.
    return get_i_datasets(count)




def visualize_rtts(df, save_figures=False, log_scale=False, swells=None, noises=None, prefix="", filetype="png", useTitle=True, bounds=(0, 0), fontsize=10):
    global features
    # Get last entry in 'id' column, aka the number of datasets to loop through.
    try:
        num = df.loc[df.index[-1], 'id']
    except:
        df['id'] = df['id'].astype(int)
        num = df.loc[df.index[-1], 'id']
    
    num = int(num)
    
    for i in range(num + 1):
        current_df = df[df['id'] == i]['rtt']
        current_df.plot(x='index', use_index=False, figsize=(16,4), linewidth=0.75, logy=log_scale) # plot the RTT

        if (useTitle):
            plt.title(f"Dataset {i:02d} RTT", loc='left')
        plt.grid(axis='y', linestyle='-', linewidth=.4) # x, y, or both
        plt.xlabel('Measurement Index')
        plt.ylabel('Round Trip Time (ms)')
        
        trait_style = 'dashed'
        xdata = list(range(len(current_df)))
        
        if swells is not None:
            if type(swells) != list:
                swells = swells.to_list()
            plt.plot(xdata, [swells[i] for x in xdata], color='orange', linestyle=trait_style, linewidth=1.5)
        if noises is not None:
            if type(noises) != list:
                noises = noises.to_list()
            plt.plot(xdata, [noises[i] for x in xdata], color='red', linestyle=trait_style, linewidth=1.5)
        
        if noises is not None or swells is not None:
            # add a legend
            plt.legend(['RTT', 'Congestion Threshold', 'Noise Threshold'], loc='upper left')
        
        if save_figures:
            if prefix != "" and not prefix.endswith("/"):
                prefix += "/"
            fig_dir = FIGURE_DIR + prefix
            if not os.path.exists(fig_dir): # create figure output dir if not already present
                os.makedirs(fig_dir)
            plt.savefig(fig_dir + f'dataset{i:02d}.{filetype}', bbox_inches='tight')
            
        plt.rc('axes', titlesize=fontsize) #fontsize of the title
        plt.rc('axes', labelsize=fontsize) #fontsize of the x and y labels
        plt.rc('xtick', labelsize=fontsize) #fontsize of the x tick labels
        plt.rc('ytick', labelsize=fontsize) #fontsize of the y tick labels
        
        plt.show()
    return




conf_matrix_dir = get_full_path('MTL', 'Figures', 'confusion matrices')

def gen_confusion_matrix(predictions, truth, index, selected_feature="", display=True, save=False, \
                         prefix="", filetype="png", dpi=100, pre="", useTitle=True, folder=""):
    """ Wrapper function for SKLearn confusion matrix metric. Returns a dictionary containing the confusion matrix values. """
    
    assert len(predictions) == len(truth)
    
    plt.rcParams["figure.dpi"] = dpi
    
    
    if type(index) is int:
        index = f"{index:02d}"
        
    matrix = confusion_matrix(truth, predictions)
    
    TN, FP, FN, TP = matrix.ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = (2 * precision * recall) / (precision + recall)
    F1 = np.nan_to_num(F1)
    
    conf_matrix = {
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "TN": TN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": F1
    }
    
    if display:
        # Heatmap implementation from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        plt.clf()
        if useTitle:
            plt.suptitle(f"MTL {index} Confusion Matrix")
        plt.title(f"Feature: {selected_feature.capitalize()}", fontsize="small")
        
#         group_names = ['True Pos','False Pos','False Neg','True Neg'] # original, wrong
        group_names = ['True Neg','False Neg','False Pos','True Pos']
        
        group_counts = ["Count: {0:0.0f}".format(value) for value in matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in matrix.flatten() / np.sum(matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        ax = sns.heatmap(matrix, annot=labels, fmt="", cmap='Blues', cbar=True, square=True, \
                         xticklabels=[0, 1], yticklabels=[0, 1], edgecolors="b")
#                    xticklabels=[1, 0], yticklabels=[1, 0], edgecolors="b")


        plt.xlabel(f'True Label\nF1 Score: {round(F1, 5)}')
        plt.ylabel('Predicted Label')
        
        for _, spine in ax.spines.items(): # add black borders around heatmap
            spine.set_visible(True)
        
        
        if folder != "":
            out_dir = f"{conf_matrix_dir}/{folder}/{selected_feature}/"
        else:
            out_dir = f"{conf_matrix_dir}/{selected_feature}/"
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        out_dir = out_dir + f"dataset_{index}.{filetype}"
        
        if save: # save figure to figures dir
            plt.savefig(out_dir, bbox_inches="tight")
        else:
            print(f"Would save figures to {out_dir}")
        plt.show()
    return conf_matrix



""" Splitting DataFrames to more easily access certain subsets. """

def get(df, index=0):
    """ Returns the data frame with the data matching the specified index. """
    return df[df['id'] == index]

def split_df(df, ct=2):
    """ Splits the specified DataFrame into ct parts. Returns a list of split DFs. """
    if len(df) % ct:
        n = len(df) % ct
        df.drop(df.tail(n).index,inplace=True) # drop last n rows to make it divisible
        
    rows = df.shape[0] # number of rows in the dataframe
    
    indices = [] # indices to split at
    remaining = math.floor(rows / ct)
    try:
        dfs = [d.reset_index() for d in np.split(df, ct)]
    except:
        dfs = [d for d in np.split(df, ct)]
    
    return dfs




""" File/dictionary serialization using pickle. """

def save(d, filename):
    """ Serialize a dictionary or object d. """
    with open(f'data/{filename}.pickle', 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
    """ Load a previously serialized object from disk. """
    with open(f'data/{filename}.pickle', 'rb') as f:
        return pickle.load(f)
    
""" File/dictionary serialization using JSON. """

def j_save(d, filename):
    """ Serialize a dictionary or list d. """
    with open(f'data/{filename}.json', 'wb') as f:
        json.dump(d, f, indent=2)

def j_load(filename):
    """ Load a previously serialized object from disk. """
    with open(f'data/{filename}.json', 'rb') as f:
        return json.load(f)
    
    