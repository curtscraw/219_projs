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
    return homogeneity_score(labels, predicted)
    
    
#returns [homogeneity, completeness, v_measure, rand, mutualr_info] for plotting 
def km_score(labels, data):
    km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
    km.fit(data)
    return print_scores(labels, km.labels_)

lsi_vals = []
nmf_vals = []
rvals = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 50, 100, 200]
print "This may take some time, be patient"
for rval in rvals:
    lsi = TruncatedSVD(n_components=rval, random_state=0)
    nmf = NMF(n_components=rval, init='random', random_state=0)
    print "With r=" + str(rval) + ":"
    print "SVD:"
    lsi_vals.append(km_score(dataset.target, lsi.fit_transform(tfidf)))
    print "NMF:"
    nmf_vals.append(km_score(dataset.target, nmf.fit_transform(tfidf)))

print "Plotting graph of homogeneity with relation to r value"
plt.plot(rvals, lsi_vals, label="SVD")
plt.plot(rvals, nmf_vals, label="NMF")
plt.xlabel('r value')
plt.ylabel('Homogeneity Score')
plt.title('Homogeneity Score relative to r value')
plt.legend(loc="lower right")
plt.show()

