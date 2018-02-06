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
import sys

if len(sys.argv) < 2:
    print "This script must be called with an r-value for reduction, for instance:"
    print "python 20_cats.py <r value>"

rval = int(sys.argv[1])
#setup, fetch datasets
this_df = 3

dataset = fetch_20newsgroups(subset='all', shuffle=True, random_state=42);

#vectorize dataset, find tfidf, report dimensions
vectorizer = text.TfidfVectorizer(min_df=this_df, stop_words='english')
tfidf = vectorizer.fit_transform(dataset.data)
print "TFxIDF dimensions: " + str(tfidf.shape)

#reduce dimensionality
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
lsi = TruncatedSVD(n_components=rval, random_state=0)
nmf = NMF(n_components=rval, init='random', random_state=0)

#cluster
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score,completeness_score,v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score,adjusted_mutual_info_score
from sklearn.metrics import confusion_matrix


def print_scores(labels, predicted):
    print "Homogeneity: " + str(homogeneity_score(labels, predicted))
    print "completeness: " + str(completeness_score(labels, predicted))
    print "V-measure: " + str(v_measure_score(labels, predicted))
    print "RAND score: " + str(adjusted_rand_score(labels, predicted))
    print "Mutual Info: " + str(adjusted_mutual_info_score(labels, predicted))
    
#returns [homogeneity, completeness, v_measure, rand, mutualr_info] for plotting 
def km_score(labels, data):
    km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
    km.fit(data)
    print_scores(labels, km.labels_)

print "With r=" + str(rval) + ":"
print "SVD:"
km_score(dataset.target, lsi.fit_transform(tfidf))
print "NMF:"
km_score(dataset.target, nmf.fit_transform(tfidf))

