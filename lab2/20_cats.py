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


svd_all = []
nmf_all = []

def print_scores(labels, predicted, svd):
    print "Homogeneity: " + str(homogeneity_score(labels, predicted))
    print "completeness: " + str(completeness_score(labels, predicted))
    print "V-measure: " + str(v_measure_score(labels, predicted))
    print "RAND score: " + str(adjusted_rand_score(labels, predicted))
    print "Mutual Info: " + str(adjusted_mutual_info_score(labels, predicted))
    ret = []
    ret.append(homogeneity_score(labels, predicted))
    ret.append(completeness_score(labels, predicted))
    ret.append(v_measure_score(labels, predicted))
    ret.append(adjusted_rand_score(labels, predicted))
    ret.append(adjusted_mutual_info_score(labels, predicted))
    if svd:
        svd_all.append(ret)
    else:
        nmf_all.append(ret)
    return homogeneity_score(labels, predicted) 
    
    
#returns [homogeneity, completeness, v_measure, rand, mutualr_info] for plotting 
def km_score(labels, data, svd):
    km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
    km.fit(data)
    return print_scores(labels, km.labels_, svd)

lsi_vals = []
nmf_vals = []
rvals = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 125, 150]
print "This may take some time, be patient"
for rval in rvals:
    lsi = TruncatedSVD(n_components=rval, random_state=0)
    nmf = NMF(n_components=rval, init='random', random_state=0)
    print "With r=" + str(rval) + ":"
    print "SVD:"
#    lsi_vals.append(km_score(dataset.target, lsi.fit_transform(tfidf), True))
    print "NMF:"
#    nmf_vals.append(km_score(dataset.target, nmf.fit_transform(tfidf), False))

print "Plotting graph of homogeneity with relation to r value"
#plt.plot(rvals, lsi_vals, label="SVD")
#plt.plot(rvals, nmf_vals, label="NMF")
#plt.xlabel('r value')
#plt.ylabel('Homogeneity Score')
#plt.title('Homogeneity Score relative to r value')
#plt.legend(loc="lower right")
#plt.show()


lsi = TruncatedSVD(n_components=80, random_state=0)
nmf = NMF(n_components=35, init='random', random_state=0)

tx_lsi = lsi.fit_transform(tfidf)
tx_nmf = nmf.fit_transform(tfidf)

print "checking if scaling or log transform improve results"
from sklearn.preprocessing import scale, FunctionTransformer
log_tf = FunctionTransformer(func=np.log, inverse_func=np.exp)
print "SVD scaled"
km_score(dataset.target, scale(tx_lsi), True)
print "NMF scaled"
km_score(dataset.target, scale(tx_nmf), True)
print "NMF log"
km_score(dataset.target, log_tf.fit_transform(tx_nmf + 0.01), True)
print "NMF log then scale"
km_score(dataset.target, scale(log_tf.fit_transform(tx_nmf + 0.01)), True)
print "NMF scale then log"
km_score(dataset.target, log_tf.fit_transform(scale(tx_nmf) + 1), True)

def cluster_plot(labels, data, xax, yax, title):
    km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
    km.fit(data)
    pred = km.predict(data)
    centers = km.cluster_centers_

    print_scores(labels, km.labels_, True)

    #plot it
    plt.scatter(data[:,0], data[:,1], c=pred, cmap='viridis')
    plt.scatter(centers[:,0], centers[:,1], c='red', s=50, alpha=0.5, marker="x")   #bigger, but 50% transparent
    plt.xlabel(xax)
    plt.ylabel(yax)
    plt.title(title)
    plt.show()
    #having some issues with graphs....
    plt.clf()
    plt.cla()
    plt.close()

cluster_plot(dataset.target, tx_lsi, "", "", "SVD clustering results")
cluster_plot(dataset.target, scale(log_tf.fit_transform(tx_nmf + 0.01)), "", "", "NMF with Log then scaling clustering results")

#used for finding best r value with searching
#print "All SVD scores:"
#for s in svd_all:
#    print s
#print "All NMF scores:"
#for s in nmf_all:
#    print s
