#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:15:51 2020

@author: kratisaxena
"""


import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
import warnings
warnings.simplefilter('ignore')
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from Init.InitPythonPaths import biorxiv_csv_path #, cord19_df_csv_path

class clustering_using_ngrams_hashvectorizer:

    def classification_report(self, model_name, test, pred):
        # function to print out classification model report
        print(model_name, ":\n")
        print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(test, pred)) * 100), "%")
        print("     Precision: ", '{:,.3f}'.format(float(precision_score(test, pred, average='micro')) * 100), "%")
        print("        Recall: ", '{:,.3f}'.format(float(recall_score(test, pred, average='micro')) * 100), "%")
        print("      F1 score: ", '{:,.3f}'.format(float(f1_score(test, pred, average='micro')) * 100), "%")

    def RandomForestClassifier_(self, X_train, y_train, X_test, y_test):

        # random forest classifier instance
        forest_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)

        # cross validation on the training set
        forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=3, n_jobs=4)

        # print out the mean of the cross validation scores
        print("Accuracy: ", '{:,.3f}'.format(float(forest_scores.mean()) * 100), "%")



        # cross validate predict on the training set
        forest_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=3, n_jobs=4)

        # print precision and recall scores
        print("Precision: ", '{:,.3f}'.format(float(precision_score(y_train, forest_train_pred, average='macro')) * 100), "%")
        print("   Recall: ", '{:,.3f}'.format(float(recall_score(y_train, forest_train_pred, average='macro')) * 100), "%")

        # first train the model
        forest_clf.fit(X_train, y_train)

        # make predictions on the test set
        forest_pred = forest_clf.predict(X_test)

        # print out the classification report
        self.classification_report("Random Forest Classifier Report (Test Set)", y_test, forest_pred)
        return forest_pred

    def classify(self, X_train, y_train, X_test, y_test):
        # we want to see how well it will classify using the labels we just created using K-Means.
        y_test_pred = self.RandomForestClassifier_(X_train, y_train, X_test, y_test)
        return y_test_pred


    def unsupervised_learning_KMeans(self, X_train, X_test, X_embedded, k=10):
        # Using K-means we will get the labels we need. For now, we will create 10 clusters. I am choosing this arbitrarily. We can change this later.

        kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)
        y_pred = kmeans.fit_predict(X_train)

        # Labels for the training set:
        y_train = y_pred

        # Labels for the test set:
        y_test = kmeans.predict(X_test)

        # sns settings
        sns.set(rc={'figure.figsize':(15,15)})

        # colors
        palette = sns.color_palette("bright", len(set(y_pred)))

        # plot
        sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
        plt.title("t-SNE Covid-19 Articles - Clustered")
        # plt.savefig("plots/t-sne_covid19_label.png")
        plt.show()
        return y_train, y_test

    def dimenstion_reduction_tsne(self, X_):
        # Using t-SNE we can reduce our high dimensional features vector into 2 dimensional plane. In the process, t-SNE will keep similar instances together while trying to push different instances far from each other. Resulting 2-D plane can be useful to see which articles cluster near each other:


        tsne = TSNE(verbose=1, perplexity=5)
        X_embedded = tsne.fit_transform(X_)


        # sns settings
        sns.set(rc={'figure.figsize':(15,15)})

        # colors
        palette = sns.color_palette("bright", 1)

        # plot
        sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)

        plt.title("t-SNE Covid-19 Articles")
        # plt.savefig("plots/t-sne_covid19.png")
        plt.show()
        return X_embedded

    def seperate_train_test(self, X):

        # test set size of 20% of the data and the random seed 42 <3
        X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)

        print("X_train size:", len(X_train))
        print("X_test size:", len(X_test), "\n")
        return X_train, X_test

    def vectorize(self, n_gram_all):
        # Now we will use HashVectorizer to create the features vector X. For now, let's limit the feature size to 2**12(4096) to speed up the computation. We might need to increase this later to reduce the collusions and improve the accuracy:

        # hash vectorizer instance
        hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)

        # features matrix X
        X = hvec.fit_transform(n_gram_all)
        return X

    def create_2_grams(self, text):
        #Let's create 2D list, where each row is instance and each column is a word. Meaning, we will separate each instance into words:
        words = []
        for ii in range(0,len(text)):
            #words.append(str(text.iloc[ii]['body_text']).split(" "))
            words.append(str(text[ii]).split(" "))
        # What we want now is n-grams from the words where n=2 (2-gram). We will still have 2D array where each row is an instance; however, each index in that row going to be a 2-gram:
        n_gram_all = []

        for word in words:
            # get n-grams for the instance
            n_gram = []
            for i in range(len(word)-2+1):
                n_gram.append("".join(word[i:i+2]))
            n_gram_all.append(n_gram)
        return n_gram_all

    def workflow(self, text):
        print("creating n-grams...")
        n_gram_all = self.create_2_grams(text)
        print("n-grams_created...")
        print("Vectorizing...")
        X = self.vectorize(n_gram_all)
        print("train-test split...")
        X_train, X_test = self.seperate_train_test(X)
        X_embedded = self.dimenstion_reduction_tsne(X_train)
        y_train, y_test = self.unsupervised_learning_KMeans(X_train, X_test, X_embedded, k=10)
        y_test_pred = self.classify(X_train, y_train, X_test, y_test)

class clustering_using_tfidf_vectors:

    def dimension_reduction_with_PCA(self, X, y):
        # t-SNE doesn't scale well.

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(X.toarray())
        # sns settings
        sns.set(rc={'figure.figsize':(15,15)})

        # colors
        palette = sns.color_palette("bright", len(set(y)))

        # plot
        sns.scatterplot(pca_result[:,0], pca_result[:,1], hue=y, legend='full', palette=palette)
        plt.title("PCA Covid-19 Articles - Clustered (K-Means) - Tf-idf with Plain Text")
        # plt.savefig("plots/pca_covid19_label_TFID.png")
        plt.show()



        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
            xs=pca_result[:,0],
            ys=pca_result[:,1],
            zs=pca_result[:,2],
            c=y,
            cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        plt.title("PCA Covid-19 Articles (3D) - Clustered (K-Means) - Tf-idf with Plain Text")
        # plt.savefig("plots/pca_covid19_label_TFID_3d.png")
        plt.show()
        return pca_result

    def dimension_reduction_with_tsne(self, X, y):

        tsne = TSNE(verbose=1)
        X_embedded = tsne.fit_transform(X.toarray())

        # sns settings
        sns.set(rc={'figure.figsize':(15,15)})

        # colors
        palette = sns.color_palette("bright", len(set(y)))

        # plot
        sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
        plt.title("t-SNE Covid-19 Articles - Clustered(K-Means) - Tf-idf with Plain Text")
        # plt.savefig("plots/t-sne_covid19_label_TFID.png")
        plt.show()
        return X_embedded

    def minibatchkmeans_(self, X, k=10):
        #Again, let's try to get our labels. We will choose 10 clusters again. This time, we will use MiniBatchKMeans as it is faster with more data:

        kmeans = MiniBatchKMeans(n_clusters=k)
        y_pred = kmeans.fit_predict(X)
        #Get the labels:

        y = y_pred
        return y

    def vectorize(self, text):
        # Vectorize Using Tf-idf with Plain Text
        vectorizer = TfidfVectorizer(max_features=2**12)
        X = vectorizer.fit_transform(text)
        return X

    def workflow(self, text):
        X = self.vectorize(text)
        y = self.minibatchkmeans_(X, k=10)
        common_obj = clustering_using_ngrams_hashvectorizer()
        X_train, X_test = common_obj.seperate_train_test(X)
        X_embedded = self.dimension_reduction_with_tsne(X, y)
        pca_result = self.dimension_reduction_with_PCA(X, y)
        X_embedded = common_obj.dimenstion_reduction_tsne(X_train)
        y_train, y_test = common_obj.unsupervised_learning_KMeans(X_train, X_test, X_embedded, k=10)
        y_test_pred = common_obj.classify(X_train, y_train, X_test, y_test)

if __name__ == "__main__":

    df = pd.read_csv(biorxiv_csv_path)
    print(df.columns)

    text = list(df['text'])

    #obj = clustering_using_ngrams_hashvectorizer()
    #obj.workflow(text)
    obj = clustering_using_tfidf_vectors()
    obj.workflow(text)

