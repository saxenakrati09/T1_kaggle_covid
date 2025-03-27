#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:08:49 2020

@author: kratisaxena
"""

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from Init.InitPythonPaths import cord19_df_csv_path
import ktrain

class LDA_TopicModelling_:

    def train_model(self, df, texts):
        ktrain.text.preprocessor.detect_lang = ktrain.text.textutils.detect_lang
        tm = ktrain.text.get_topic_model(texts, n_topics=None, n_features=10000)
        tm.print_topics()
        tm.build(texts, threshold=0.25)

        texts = tm.filter(texts)
        df = tm.filter(df)
        tm.visualize_documents(doc_topics=tm.get_doctopics())
        return texts, df, tm

    def get_topics(self, tm, keyword):
        keyword_results = tm.search(keyword, case_sensitive=False)
        threshold = .80
        keyword_topic_ids = {doc[3] for doc in keyword_results if doc[2]>threshold}
        t_topics = keyword_topic_ids.copy()

        docs = tm.get_docs(topic_ids=t_topics, rank=True)
        print("TOTAL_NUM_OF_DOCS: %s" % len(docs))

        print("##################################")

        for t in t_topics:
            docs = tm.get_docs(topic_ids=[t], rank=True)
            print("NUM_OF_DOCS: %s" % len(docs))
            if(len(docs)==0): continue
            doc = docs[1]
            print('TOPIC_ID: %s' % (doc[3]))
            print('TOPIC: %s' % (tm.topics[t]))
            print('DOC_ID: %s'  % (doc[1]))
            print('TOPIC SCORE: %s '% (doc[2]))
            print('TEXT: %s' % (doc[0][0:400]))


            print("##################################")
        tm.train_recommender()
        return tm

if __name__ == "__main__":
    obj = LDA_TopicModelling_()
    df = pd.read_csv(cord19_df_csv_path)
    texts = df["body_text"]
    texts, df, tm = obj.train_model(df, texts)
    tm = obj.get_topics(tm, "transmission")
    text = "What is known about covid-19 transmission?"
    for i, doc in enumerate(tm.recommend(text=text, n=5)):
        print('RESULT #%s'% (i+1))
        print('TEXT:\n\t%s' % (" ".join(doc[0].split()[:500])))
        print()

    df = pd.read_csv(cord19_df_csv_path)
    texts = df["body_text"]
    texts, df, tm = obj.train_model(df, texts)
    tm = obj.get_topics(tm, "incubation")
    text = "What is known about covid-19 incubation period?"
    for i, doc in enumerate(tm.recommend(text=text, n=5)):
        print('RESULT #%s'% (i+1))
        print('TEXT:\n\t%s' % (doc[0]))
        print()

    df = pd.read_csv(cord19_df_csv_path)
    texts = df["body_text"]
    texts, df, tm = obj.train_model(df, texts)
    tm = obj.get_topics(tm, "environmental stability")
    text = "What is known about covid-19 environmental stability?"
    for i, doc in enumerate(tm.recommend(text=text, n=5)):
        print('RESULT #%s'% (i+1))
        print('TEXT:\n\t%s' % (doc[0]))
        print()

    '''
    ktrain.text.preprocessor.detect_lang = ktrain.text.textutils.detect_lang

    tm = ktrain.text.get_topic_model(texts, n_topics=None, n_features=10000)
    tm.print_topics()
    tm.build(texts, threshold=0.25)

    texts = tm.filter(texts)
    df = tm.filter(df)
    tm.visualize_documents(doc_topics=tm.get_doctopics())

    # What is known about transmission, incubation, and environmental stability? *
    transmission_results = tm.search('transmission', case_sensitive=False)
    incubation_results = tm.search('incubation', case_sensitive=False)
    environmental_results = tm.search('environmental stability', case_sensitive=False)

    threshold = .80

    transmission_topic_ids = {doc[3] for doc in transmission_results if doc[2]>threshold}
    incubation_topic_ids = {doc[3] for doc in incubation_results if doc[2]>threshold}
    environmental_topic_ids = {doc[3] for doc in environmental_results if doc[2]>threshold}

    t_topics = transmission_topic_ids.copy()
    t_topics.update(incubation_topic_ids)
    t_topics.update(environmental_topic_ids)

    tm.visualize_documents(doc_topics=tm.get_doctopics(t_topics))

    docs = tm.get_docs(topic_ids=t_topics, rank=True)
    print("TOTAL_NUM_OF_DOCS: %s" % len(docs))

    print("##################################")

    for t in t_topics:
        docs = tm.get_docs(topic_ids=[t], rank=True)
        print("NUM_OF_DOCS: %s" % len(docs))
        if(len(docs)==0): continue
        doc = docs[1]
        print('TOPIC_ID: %s' % (doc[3]))
        print('TOPIC: %s' % (tm.topics[t]))
        print('DOC_ID: %s'  % (doc[1]))
        print('TOPIC SCORE: %s '% (doc[2]))
        print('TEXT: %s' % (doc[0][0:400]))


        print("##################################")

    tm.train_recommender()

    text = "What is known about covid-19 transmission?"
    for i, doc in enumerate(tm.recommend(text=text, n=5)):
        print('RESULT #%s'% (i+1))
        print('TEXT:\n\t%s' % (" ".join(doc[0].split()[:500])))
        print()

    text = "What is known about covid-19 incubation period?"
    for i, doc in enumerate(tm.recommend(text=text, n=5)):
        print('RESULT #%s'% (i+1))
        print('TEXT:\n\t%s' % (doc[0]))
        print()

    text = "What is known about covid-19 environmental stability?"
    for i, doc in enumerate(tm.recommend(text=text, n=5)):
        print('RESULT #%s'% (i+1))
        print('TEXT:\n\t%s' % (doc[0]))
        print()
    '''