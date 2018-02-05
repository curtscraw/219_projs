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

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42);

#function to de-generalize to comp-tech or recreation
def class_combine(in_targets):
    out = []
    for i in in_targets:
        if i <= 3:
            out.append(0)
        else:
            out.append(1)

    return (out, ('comp-tech', 'rec'))
print "Relabeling data with either 0 for comp-tech or 1 for rec"
labels, label_names = class_combine(dataset.target)

#Part 1
#vectorize dataset, find tfidf, report dimensions
vectorizer = text.TfidfVectorizer(min_df=this_df, stop_words='english')
tfidf = vectorizer.fit_transform(dataset.data)
print "TFxIDF dimensions: " + str(tfidf.shape)

#Part 2
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score,completeness_score,v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score,adjusted_mutual_info_score
km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)

print "Clustering sparse tfidf with kmeans, k=2"
km.fit(tfidf)

print "Homogeneity: " + str(homogeneity_score(labels, km.labels_))
print "completeness: " + str(completeness_score(labels, km.labels_))
print "V-measure: " + str(v_measure_score(labels, km.labels_))
print "RAND score: " + str(adjusted_rand_score(labels, km.labels_))
print "Mutual Info: " + str(adjusted_mutual_info_score(labels, km.labels_))

#part 3
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
#TODO

#part 4
#TODO
