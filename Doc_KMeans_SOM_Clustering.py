#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:37:09 2020

@author: kratisaxena
"""

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

import gensim
from gensim.models import Doc2Vec

from sklearn.cluster import KMeans

from Init.InitPythonPaths import root_path, metadata_path

def dealing_with_null_values(dataset):
    dataset = dataset
    for i in dataset.columns:
        replace = []
        data  = dataset[i].isnull()
        count = 0
        for j,k in zip(data,dataset[i]):
            if (j==True):
                count = count+1
                replace.append('No Information Available')
            else:
                replace.append(k)
        print("Num of null values (",i,"):",count)
        dataset[i] = replace
    return dataset

# Doc2Vec for Text Representation and KMeans Clustering
class Doc2vec_clustering:

    def get_matrix(self, df):

        document_tagged = []
        tagged_count = 0
        for _ in df['abstract'].values:
            document_tagged.append(gensim.models.doc2vec.TaggedDocument(_,[tagged_count]))
            tagged_count += 1
        print(tagged_count)
        d2v = Doc2Vec(document_tagged)
        d2v.train(document_tagged,epochs=d2v.epochs,total_examples=d2v.corpus_count)
        return d2v

    def cluster_with_varying_numbers(self, d2v):
        #cluster 5 and cluster 6 models
        X = np.array(d2v.docvecs.vectors_docs)
        kmeans5 = KMeans(n_clusters = 5,random_state=0)
        km5 = kmeans5.fit_predict(X)
        kmeans6 = KMeans(n_clusters = 6,random_state=0)
        km6 = kmeans6.fit_predict(X)
        kmeans7 = KMeans(n_clusters = 7,random_state=0)
        km7 = kmeans7.fit_predict(X)
        kmeans8 = KMeans(n_clusters = 8,random_state=0)
        km8 = kmeans8.fit_predict(X)
        kmeans9 = KMeans(n_clusters = 9,random_state=0)
        km9 = kmeans9.fit_predict(X)
        kmeans10 = KMeans(n_clusters = 10,random_state=0)
        km10 = kmeans10.fit_predict(X)
        kmeans15 = KMeans(n_clusters = 15,random_state=0)
        km15 = kmeans15.fit_predict(X)
        kmeans20 = KMeans(n_clusters = 20,random_state=0)
        km20 = kmeans20.fit_predict(X)
        kmeans25 = KMeans(n_clusters = 25,random_state=0)
        km25 = kmeans25.fit_predict(X)
        models = [kmeans5, kmeans6, kmeans7,kmeans8,kmeans9,kmeans10, kmeans15, kmeans20, kmeans25]
        return models

    def plot_WCSS_BCSS(self, models, data):
        fig, ax = plt.subplots(1, 2, figsize=(12,5))

        ## Plot WCSS
        wcss = [mod.inertia_ for mod in models]
        n_clusts = [5,6,7,8,9,10,15,20,25]

        ax[0].bar(n_clusts, wcss,color = 'red')
        ax[0].set_xlabel('Number of clusters')
        ax[0].set_ylabel('WCSS')
        ax[0].set_title('Within Cluster Analysis')


        ## Plot BCSS
        n_1 = (float(data.shape[0]) * float(data.shape[1])) - 1.0
        tss = n_1 * np.var(data)
        bcss = [tss - x for x in wcss]
        ax[1].bar(n_clusts, bcss)
        ax[1].set_xlabel('Number of clusters')
        ax[1].set_ylabel('BCSS')
        ax[1].set_title('Between Cluster Analysis')
        plt.show()

if __name__ == "__main__":
    # Data Preprocessing
    #1
    #read data set
    meta_data = pd.read_csv(metadata_path)
    print('Original Size of Data:',meta_data.shape)

    #drop rows with null values (based on abstract attribute)
    meta_data.dropna(subset = ['abstract'],axis = 0, inplace = True)
    print('Data Size after dropping rows with null values (based on abstract attribute):',meta_data.shape)

    #2
    #handling duplicate data (based on 'sha','title' and 'abstract')
    print(meta_data[meta_data.duplicated(subset=['sha','title','abstract'], keep=False) == True])
    meta_data.drop_duplicates(subset=['sha','title','abstract'],keep ='last',inplace=True)
    print('Data Size after dropping duplicated data (based on abstract attribute):',meta_data.shape)

    meta_data = dealing_with_null_values(meta_data)




    plot_WCSS_BCSS(models,X)