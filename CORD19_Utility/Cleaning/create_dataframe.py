#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:36:25 2020

@author: kratisaxena
"""


import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

import glob
import json
#import scispacy
import spacy
#import en_core_sci_md
from spacy_langdetect import LanguageDetector

from Init.InitPythonPaths import root_path, metadata_path, cord19_df_csv_path

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

if __name__ == "__main__":
    meta_df = pd.read_csv(metadata_path, dtype={
        'pubmed_id': str,
        'Microsoft Academic Paper ID': str,
        'doi': str
    })
    meta_df.head(2)

    all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
    print(len(all_json))


    first_row = FileReader(all_json[0])
    print(first_row)

    dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}
    for idx, entry in enumerate(all_json):
        if idx % (len(all_json) // 10) == 0:
            print(f'Processing index: {idx} of {len(all_json)}')
        content = FileReader(entry)
        dict_['paper_id'].append(content.paper_id)
        dict_['abstract'].append(content.abstract)
        dict_['body_text'].append(content.body_text)
    papers = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])
    print(papers.head())
    df = pd.merge(papers, meta_df, left_on='paper_id', right_on='sha', how='left').drop('sha', axis=1)

    # some new columns for convenience
    df['publish_year'] = df.publish_time.str[:4].fillna(-1).astype(int) # 360 times None

    # Exploration and Cleaning
    '''
    print("\nDifferent Abstract in Metadata and JSON files")

    print(df[df.abstract_x != df.abstract_y].shape)
    print(df[df.abstract_x != df.abstract_y].head())

    print("\nChecking some of the files online, it seems that where the abstract is missing in the metadata, the abstract in the JSON file is simply the beginning of the text.")
    print(df[df.abstract_x != df.abstract_y][['abstract_x', 'abstract_y', 'url']][
    (df.abstract_y.isnull()) & (df.abstract_x != '') & (~df.url.isnull())])

    print("\nmissing abstracts in json files")
    print(df.abstract_x.isnull().sum())
    print((df.abstract_x =='').sum()) # missing abstracts in json files

    print("\nmissing abstracts in metadata")
    print(df.abstract_y.isnull().sum())
    print((df.abstract_y=='').sum()) # missing abstracts in metadata
    '''
    # Since the abstracts from the metadata seem more reliable we generally use these, but fill the missing values with the abstract from the extracted values from the JSON file.

    df.loc[df.abstract_y.isnull() & (df.abstract_x != ''), 'abstract_y'] = df[(df.abstract_y.isnull()) & (df.abstract_x != '')].abstract_x
    df.rename(columns = {'abstract_y': 'abstract'}, inplace=True)
    df.drop('abstract_x', axis=1, inplace=True)

    # drop duplicates
    df.drop_duplicates(['paper_id', 'body_text'], inplace=True)
    print("\ndf created")
    '''
    from IPython.utils import io

    with io.capture_output() as captured:
        !pip install scispacy
        !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz
        !pip install spacy-langdetect
        !pip install spac scispacy spacy_langdetect https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.3/en_core_sci_lg-0.2.3.tar.gz
    '''

    # medium model
    nlp = spacy.load("en_core_web_sm")
    #nlp = en_core_sci_md.load(disable=["tagger", "ner"])
    nlp.max_length = 2000000
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    print("finding articles in english language")
    df['text_language'] = df.body_text.apply(lambda x: nlp(str(x[:2000]))._.language['language'])

    df.text_language.value_counts()
    df.loc[df[df.text_language != 'en'].index].shape
    df = df.drop(df[df.text_language != 'en'].index)
    df.to_csv(cord19_df_csv_path, index=False)
