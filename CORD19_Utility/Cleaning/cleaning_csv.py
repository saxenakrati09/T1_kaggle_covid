#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:00:40 2020

@author: kratisaxena
"""


import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from Init.InitPythonPaths import biorxiv_csv_path, pmc_csv_path, comm_csv_path, noncomm_csv_path, biorxiv_cleaner_csv_path, pmc_cleaner_csv_path, comm_cleaner_csv_path, noncomm_cleaner_csv_path, all_papers_cleaner_path

# Data Cleaning
def clean_up(t):
    """
    Cleans up the passed value
    -Remove newline characters
    -Remove citation numbers like [5]
    -Remove et. al
    -Remove Fig and Table citations like (Fig 5) or ( Table 6 )
    -Replace continuous spaces with a single space
    -Convert all alphabets to lower case

    """
    # Remove New Lines
    t = t.replace("\n"," ") # removes newlines

    # Remove citation numbers (Eg.: [4])
    t = re.sub("\[[0-9]+(, [0-9]+)*\]", "", t)

    # Remove et al.
    t = re.sub("et al.", "", t)

    # Remove Fig and Table
    t = re.sub("\( ?Fig [0-9]+ ?\)", "", t)
    t = re.sub("\( ?Table [0-9]+ ?\)", "", t)

    # Replace continuous spaces with a single space
    t = re.sub(' +', ' ', t)

    # Convert all to lowercase
    t = t.lower()

    return t

def clean_dataframes(bio, noncomm, comm, pmc):
    # Impute NaNs with "Missing"

    bio = bio.fillna("Missing")
    noncomm = noncomm.fillna("Missing")
    comm = comm.fillna("Missing")
    pmc = pmc.fillna("Missing")

    # Concatenate all the dataframes together
    papers = pd.concat([bio, comm, noncomm, pmc], ignore_index=True)

    papers["abstract"] = papers["abstract"].apply(clean_up)
    papers["text"] = papers["text"].apply(clean_up)

    return bio, noncomm, comm, pmc, papers

def plot_missing_values(bio, noncomm, comm, pmc):
    # Missing Value Visualization

    fig, axes = plt.subplots(2 ,2, figsize=(12, 6))

    title = "Biorxiv"
    ax = sns.heatmap(bio.isnull(), cmap="Reds", cbar=False, ax=axes[0,0])
    ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
    axes[0,0].set_title(title, fontsize=15)

    title = "Non-commercial Use"
    ax = sns.heatmap(noncomm.isnull(), cmap="Reds", cbar=False, ax=axes[0,1])
    ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
    axes[0,1].set_title(title, fontsize=15)

    title = "Commercial Use"
    ax = sns.heatmap(comm.isnull(), cmap="Reds", cbar=False, ax=axes[1,0])
    ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
    axes[1,0].set_title(title, fontsize=15)

    title = "PMC"
    ax = sns.heatmap(pmc.isnull(), cmap="Reds", cbar=False, ax=axes[1,1])
    ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
    axes[1,1].set_title(title, fontsize=15)

    fig.suptitle("Missing Value Heatmaps for all 4 datasets", fontsize=20)
    plt.show()

if __name__ == "__main__":
    # Load the datasets
    bio = pd.read_csv(biorxiv_csv_path)
    pmc = pd.read_csv(pmc_csv_path)
    comm = pd.read_csv(comm_csv_path)
    noncomm = pd.read_csv(noncomm_csv_path)
    plot_missing_values(bio, noncomm, comm, pmc)
    bio, noncomm, comm, pmc, papers = clean_dataframes(bio, noncomm, comm, pmc)

    bio.to_csv(biorxiv_cleaner_csv_path, index=False)
    noncomm.to_csv(noncomm_cleaner_csv_path, index=False)
    comm.to_csv(comm_cleaner_csv_path, index=False)
    pmc.to_csv(pmc_cleaner_csv_path, index=False)
    papers.to_csv(all_papers_cleaner_path, index=False)
