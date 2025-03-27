#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:22:58 2020

@author: kratisaxena
"""

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cord19q.etl import Etl

# Build SQLite database for metadata.csv and json full text files
Etl.run("/Data/CORD-19-research-challenge", "cord19q")

from cord19q.index import Index

# Build the embeddings index
Index.run("cord19q", "/Data/cord19-fasttext-vectors/cord19-300d.magnitude")