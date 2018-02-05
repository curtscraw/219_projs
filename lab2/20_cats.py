"""
Curtis Crawford, 805024638
Abdullah-Al-Zubaer Imran, 804733867

EE219, Winter 2018
Project 2
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
import matplotlib as mpl
import matplotlib.pyplot as plt

#setup, fetch datasets
this_df = 3

categories = None

dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42);

#vectorize dataset, find tfidf, report dimensions
vectorizer = text.TfidfVectorizer(min_df=this_df, stop_words='english')
tfidf = vectorizer.fit_transform(dataset.data)
print "TFxIDF dimensions: " + str(tfidf.shape)

#reduce dimensionality

#cluster
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score,completeness_score,v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score,adjusted_mutual_info_score
km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
