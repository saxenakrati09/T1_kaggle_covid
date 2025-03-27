#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:49:18 2020

@author: kratisaxena
"""

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
import json
from copy import deepcopy

import pandas as pd
from tqdm.notebook import tqdm

from Init.InitPythonPaths import biorxiv_dir, pmc_dir, comm_dir, noncomm_dir, biorxiv_csv_path, pmc_csv_path, comm_csv_path, noncomm_csv_path

def format_name(author):
    middle_name = " ".join(author['middle'])

    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))

    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []

    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)

    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}

    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"

    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []

    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'],
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)

def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = os.path.join(dirname, filename)
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)

    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []

    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'],
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text',
                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()

    return clean_df

if __name__ == "__main__":
    #biorxiv_dir = '/Users/kratisaxena/Desktop/Jupyter_notebooks/Kaggle_Covid/CORD19_Utility/Data/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'
    filenames = os.listdir(biorxiv_dir)
    print("Number of articles retrieved from biorxiv:", len(filenames))

    all_files = []

    for filename in filenames:
        filename = os.path.join(biorxiv_dir, filename)
        file = json.load(open(filename, 'rb'))
        all_files.append(file)
    '''
    file = all_files[0]
    print("Dictionary keys:", file.keys())

    # Abstract
    # Let's see what the grouped section titles are for the example above
    texts = [(di['section'], di['text']) for di in file['body_text']]
    texts_di = {di['section']: "" for di in file['body_text']}
    for section, text in texts:
        texts_di[section] += text

    # The following example shows what the final result looks like, after we format each section title with its content:
    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"

    # Metadata
    authors = all_files[0]['metadata']['authors']

    # The format_name and format_affiliation functions:
    authors = all_files[4]['metadata']['authors']
    print("Formatting without affiliation:")
    print(format_authors(authors, with_affiliation=False))
    print("\nFormatting with affiliation:")
    print(format_authors(authors, with_affiliation=True))

    # Bibliography
    bibs = list(file['bib_entries'].values())

    # You can reused the format_authors function here:
    format_authors(bibs[1]['authors'], with_affiliation=False)
    bib_formatted = format_bib(bibs[:5])
    '''
    # Generate csv for BIOrxiv
    cleaned_files = []

    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'],
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = [
    'paper_id',
    'title',
    'authors',
    'affiliations',
    'abstract',
    'text',
    'bibliography',
    'raw_authors',
    'raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()

    clean_df.to_csv(biorxiv_csv_path, index=False)

    #Generate CSV: Custom (PMC), Commercial, Non-commercial licenses

    #pmc_dir = '/Users/kratisaxena/Desktop/Jupyter_notebooks/Kaggle_Covid/CORD19_Utility/Data/CORD-19-research-challenge/custom_license/custom_license/'
    pmc_files = load_files(pmc_dir)
    pmc_df = generate_clean_df(pmc_files)
    pmc_df.to_csv(pmc_csv_path, index=False)

    #comm_dir = '/Users/kratisaxena/Desktop/Jupyter_notebooks/Kaggle_Covid/CORD19_Utility/Data/CORD-19-research-challenge/comm_use_subset/comm_use_subset/'
    comm_files = load_files(comm_dir)
    comm_df = generate_clean_df(comm_files)
    comm_df.to_csv(comm_csv_path, index=False)

    #noncomm_dir = '/Users/kratisaxena/Desktop/Jupyter_notebooks/Kaggle_Covid/CORD19_Utility/Data/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/'
    noncomm_files = load_files(noncomm_dir)
    noncomm_df = generate_clean_df(noncomm_files)
    noncomm_df.to_csv(noncomm_csv_path, index=False)