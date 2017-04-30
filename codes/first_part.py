#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:19:27 2017

@author: Jingyi
"""

import pandas as pd
import os

os.chdir("/Users/Jingyi/Desktop/CS 289A/final_project/yelp_dataset_challenge_round9")

review = pd.read_csv("yelp_academic_dataset_review.csv")
business = pd.read_csv("yelp_academic_dataset_business.csv")

business_nc = business[business.state == "b'NC'"]
nc_bus_id = list(business_nc["business_id"])

## get all the reviews for business in North Carolina
review_nc = review[review.business_id.isin(nc_bus_id)]
star_nc = list(review_nc.stars)
text_nc = list(review_nc.text)

import numpy as np
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest,chi2
# import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

text_nc_array = [text_nc[i][2:-1] for i in range(len(text_nc))]

vectorizer = TfidfVectorizer(ngram_range = (1,1),analyzer="word", lowercase=False)
text_features = vectorizer.fit_transform(text_nc_array)
vocab = vectorizer.get_feature_names()

#text_features = vectorizer.transform(text_nc_array)
fselect = SelectKBest(chi2 , k=1700)
text_features = fselect.fit_transform(text_features,star_nc)
text_features = text_features.toarray()

import sklearn
text_features, star_nc = sklearn.utils.shuffle(text_features, star_nc)

train