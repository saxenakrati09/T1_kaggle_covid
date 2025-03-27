#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:56:18 2020

@author: kratisaxena
"""

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
import warnings
warnings.simplefilter('ignore')
import pandas as pd

from Init.InitPythonPaths import bert_mean_token_path, biorxiv_csv_path
from sentence_transformers import SentenceTransformer
import scipy

class QandA_:

    def load_model(self, corpus, NLP_MODEL):
        NLP_MODEL = bert_mean_token_path
        embedder = SentenceTransformer(NLP_MODEL)

        corpus_embeddings = embedder.encode(corpus)

        print("corpus embeddings made")
        return embedder, corpus_embeddings

    def ask_question(self, query, corpus, corpus_embeddings, embedder):
        # inputs text query and results top 5 matching answers
        queries = [query]
        query_embeddings = embedder.encode(queries)

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        closest_n = 5
        for query, query_embedding in zip(queries, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])
            #display(Markdown('## Question -> %s'%query))
            print('## Question -> %s'%query)
            #display(Markdown('**Top 5 answers compiled below by running AI algorithm on research text.**<hr>'))
            print('**Top 5 answers compiled below by running AI algorithm on research text.**<hr>')

            # get the closest answers
            for idx, distance in results[0:closest_n]:
                #display(Markdown('- ### ' + corpus[idx].strip() + " (Score: %.4f)" % (1-distance)))
                print('- ### ' + corpus[idx].strip() + " (Score: %.4f)" % (1-distance))
            #display(Markdown('<hr>'))
            print("\n")


if __name__ == "__main__":
    df = pd.read_csv(biorxiv_csv_path)
    corpus = list(df['text']) #[:2]
    print("Corpus size = %d"%(len(corpus)))

    NLP_MODEL = bert_mean_token_path
    obj = QandA_()
    embedder, corpus_embeddings = obj.load_model(corpus, NLP_MODEL)
    print("corpus embeddings made")

    obj.ask_question('What is the reproduction rate of coronavirus?', corpus, corpus_embeddings, embedder)
