#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:17:32 2020

@author: kratisaxena
"""


import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load libraries
import pandas as pd

from Init.InitPythonPaths import all_papers_cleaner_path


class keyword_search_:

    '''
    Finding the Necessary Papers based on simple "Keyword" search
    '''
    '''
    The biggest dilemma is to select the right kind of papers to get the right information pertaining to a given task. This section deals with re-usable code that can help find the right papers based on specific keywords we are looking for.

How to use this section?

Step 1. Choose your dataframe (You can either choose one of the 4 initial dataframes loaded or select papers which consists of all papers (I will be using the latter in this notebook)
Step 2. Choose a specific keyword you are looking for (Eg. :smoking)
Step 3. Run the select_papers() function with necessary arguments
The original dataframe will now have 3 more features, each feature depicting the presence of the given keyword in title, abstract or text respectively.
    '''
    def word_occurence(self, entry, word):
        """
        Identifies if a given word exists in a dataframe's entry
        or not

        Parameters
        ----------
        entry : The entry OR cell or value

        Returns
        -------
        0 if the word is not found
        1 if it the word is found
        """
        # convert to lower case for uniformity
        word=word.lower()

        if(word in entry.lower()):
            return 1
        else:
            return 0

    def select_papers(self, dataframe, keyword):
        """
        Creates new features in a dataframe depicting
        the existence of the given keyword

        Parameters
        ----------
        dataframe : The dataframe in which you are searching
                    for
        keyword : The keyword that you are searching for

        Returns
        -------
        The new dataframe with the newly created
        feature/columns
        """
        '''
        NOTE : Essentially what the select_papers() function is doing is Introducing 3 new features everytime for every new word that is being searched for. These features correspond to the presence of a given word in

Title of the Paper
Abstract of the Paper
Text of the Paper If the word exists, it is represented with a 1. Else, it's a 0.
        '''
        # title
        feature_header = keyword+"_exists_title"
        dataframe[feature_header] = dataframe["title"].apply(self.word_occurence, word=keyword)

        # abstract
        feature_header = keyword+"_exists_abstract"
        dataframe[feature_header] = dataframe["abstract"].apply(self.word_occurence, word=keyword)

        # text
        feature_header = keyword+"_exists_text"
        dataframe[feature_header] = dataframe["text"].apply(self.word_occurence, word=keyword)

        return dataframe

class analyzing_papers_:
    '''
    Analyzing Papers that contain words like "pulmonary" or "smoking"
    '''
    def display_papers(self, dataframe):
        """
        Displays all the papers in a
        data subset obtained like
        bio_pulmonary or bio_smoking

        Parameters
        ----------
        dataframe : The dataframe

        Returns
        -------
        Prints all paper titles and paper ids
        in a given dataframe
        """
        papers = ";".join(comment for comment in dataframe["title"])
        paper_ids = ";".join(comment for comment in dataframe["paper_id"])
        papers = papers.split(";")
        paper_ids = paper_ids.split(";")
        for p,p_id in zip(papers, paper_ids):
            print("-> ",p," ( Paper ID :", p_id,")")
        print("----------")


    def choosing_sentences(self, text, word):
        """
        Function to choose sentences from a text
        passage/string based on the existence of a
        given word in these strings

        Parameters
        ----------
        text : The text passage
        word : Word to search for in the text

        Returns
        -------
        Sentences that contain the word
        """
        # Initializing empty list
        qualified = []

        text = text.split(".")
        text = [x.strip() for x in text]
        for i in text:
            if(word in i):
                qualified.append(i)
        return qualified

    def extract_impt_sentences(self, dataframe, identifier, feature_list, keyword):
        """
        Select important sentences from textual features that
        contain a specific keyword

        Parameters
        ----------
        dataframe : The Dataframe
        identifier : The feature that has the ability
                     to uniquely identify each row
        feature_list : The features that are to be
                       searched for the given keyword
        keyword : The keyword to search for; every sentence
                  having this keyword should be stored in a
                  list and returned

        Returns
        -------
        Dictionary with
        Keys : Paper titles
        Values : Sentences that have the word in the given paper
        """
        # number of rows
        #n_rows = dataframe.shape[0]
        documents = {}

        for f in feature_list:
            for i in range(dataframe.shape[0]):
                u_id = dataframe[identifier].iloc[i]
                qualifying_sentences = self.choosing_sentences(dataframe[f].iloc[i], keyword)
                # ignore papers where there are NO QUALIFYING SENTENCES
                if(len(qualifying_sentences) != 0):
                    documents[u_id] = qualifying_sentences
                else:
                    pass
        return (documents)

    def choose_sentences_based_on_words(self, list_sentences, list_words):
        """
        Chooses all sentences in the list_sentences that consist of
        words from list_words
        """
        q = []
        for i in list_sentences:
            flag=0
            for j in list_words:
                if(not(j in i)):
                    flag+=1
            if(flag==0):
                q.append(i)
        return q

class keyword_search_workflow_:

    def keyword_search_workflow_single_keyword(self, keyword, papers):
        ks_obj = keyword_search_()
        papers = ks_obj.select_papers(papers, keyword)
        # subsetting
        papers_keyword = papers[(papers[keyword + "_exists_title"]==1) | (papers[keyword + "_exists_abstract"]==1) | (papers[keyword + "_exists_text"]==1)]

        ana_obj = analyzing_papers_()
        print("\n10 Papers in which the word "+ keyword +" is mentioned :\n")
        ana_obj.display_papers(papers_keyword.sample(10))
        return papers_keyword

    def keyword_search_workflow_two_keyword(self, keyword1, keyword2, papers):
        ana_obj = analyzing_papers_()
        papers_keyword = self.keyword_search_workflow_single_keyword(keyword1, papers)
        keyword_docs = ana_obj.extract_impt_sentences(papers_keyword, "title", ["abstract", "text"], keyword1)
        keyword_docs_risk = {}
        for u_id,sent in keyword_docs.items():
            s = ana_obj.choose_sentences_based_on_words(sent, [keyword2, "cov-"])
            if(len(s)!=0):
                keyword_docs_risk[u_id] = s

        for u_id,sent in keyword_docs.items():
            s = ana_obj.choose_sentences_based_on_words(sent, [keyword2, "-cov"])
            if(len(s)!=0):
                keyword_docs_risk[u_id] = s

        for u_id,sent in keyword_docs.items():
            s = ana_obj.choose_sentences_based_on_words(sent, [keyword2, "hcov"])
            if(len(s)!=0):
                keyword_docs_risk[u_id] = s

        for u_id,sent in keyword_docs.items():
            s = ana_obj.choose_sentences_based_on_words(sent, [keyword2, "coronavirus"])
            if(len(s)!=0):
                keyword_docs_risk[u_id] = s

        for u_id,sent in keyword_docs.items():
            s = ana_obj.choose_sentences_based_on_words(sent, [keyword2, "covid"])
            if(len(s)!=0):
                keyword_docs_risk[u_id] = s

        for u_id,sentence in keyword_docs_risk.items():
            print("Paper Title : "+u_id)
            print(sentence)
            print()

    def related_keyword_search_multi_words(self, list_of_keywords, related_keyword,keyword2, papers):
        # Selecting papers
        ks_obj = keyword_search_()
        ana_obj = analyzing_papers_()
        for i in range(len(list_of_keywords)):
            papers = ks_obj.select_papers(papers, list_of_keywords[i])
        papers_rel_keywords = papers
        for i in range(len(list_of_keywords)):
            papers_rel_keywords = papers_rel_keywords[(papers_rel_keywords[list_of_keywords[i] + "_exists_title"]==1) | (papers_rel_keywords[list_of_keywords[i] + "_exists_abstract"]==1) | (papers_rel_keywords[list_of_keywords[i] + "_exists_text"]==1)]

        print("10 Papers in which "+ str(list_of_keywords) +" are mentioned atleast once:\n")
        ana_obj.display_papers(papers_rel_keywords.sample(10))

        keyword_docs = ana_obj.extract_impt_sentences(papers_rel_keywords, "title", ["abstract", "text"], related_keyword) # to relate to any pregnan- word
        keyword_docs_risk = {}
        for u_id,sent in keyword_docs.items():
            s = ana_obj.choose_sentences_based_on_words(sent, [keyword2, "-cov"])
            if(len(s)!=0):
                keyword_docs_risk[u_id] = s

        for u_id,sent in keyword_docs.items():
            s = ana_obj.choose_sentences_based_on_words(sent, [keyword2, "cov-"])
            if(len(s)!=0):
                keyword_docs_risk[u_id] = s

        for u_id,sent in keyword_docs.items():
            s = ana_obj.choose_sentences_based_on_words(sent, [keyword2, "hcov"])
            if(len(s)!=0):
                keyword_docs_risk[u_id] = s

        for u_id,sent in keyword_docs.items():
            s = ana_obj.choose_sentences_based_on_words(sent, [keyword2, "coronavirus"])
            if(len(s)!=0):
                keyword_docs_risk[u_id] = s

        for u_id,sent in keyword_docs.items():
            s = ana_obj.choose_sentences_based_on_words(sent, [keyword2, "covid"])
            if(len(s)!=0):
                keyword_docs_risk[u_id] = s

        for u_id,sentence in keyword_docs_risk.items():
            print("Paper Title : "+u_id)
            print(sentence)
            print()


if __name__ == "__main__":
    papers = pd.read_csv(all_papers_cleaner_path)
    ksw_obj = keyword_search_workflow_()
    ksw_obj.keyword_search_workflow_single_keyword("smoking", papers)
    ksw_obj.keyword_search_workflow_two_keyword("smoking", "risk", papers)
    ksw_obj.related_keyword_search_multi_words(['women', 'pregnancy', 'pregnant', 'newborn', 'neonate'], 'pregnan','risk', papers)
    ksw_obj.related_keyword_search_multi_words(['women', 'pregnancy', 'pregnant', 'newborn', 'neonate'], 'newborn','risk', papers)
    ksw_obj.related_keyword_search_multi_words(['women', 'pregnancy', 'pregnant', 'newborn', 'neonate'], 'neonate','risk', papers)
