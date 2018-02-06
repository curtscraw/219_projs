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

print "Relabeling data with 0 for comp-tech or 1 for rec"
labels, label_names = class_combine(dataset.target)

#vectorize dataset, find tfidf, report dimensions
vectorizer = text.TfidfVectorizer(min_df=this_df, stop_words='english')
tfidf = vectorizer.fit_transform(dataset.data)

#Dimension reduction classes
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
print "Using r=25 for SVD and r=10 for NMF"
lsi = TruncatedSVD(n_components=25)
nmf = NMF(n_components=10)
tfidf = tfidf

#plot the two scatter plots
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score,completeness_score,v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score,adjusted_mutual_info_score
from sklearn.metrics import confusion_matrix

def print_scores(labels, predicted):
    print "Contingency: "
    print str(confusion_matrix(labels, predicted))
    print "Homogeneity: " + str(homogeneity_score(labels, predicted))
    print "completeness: " + str(completeness_score(labels, predicted))
    print "V-measure: " + str(v_measure_score(labels, predicted))
    print "RAND score: " + str(adjusted_rand_score(labels, predicted))
    print "Mutual Info: " + str(adjusted_mutual_info_score(labels, predicted))

def cluster_plot(labels, data, xax, yax, title):
    km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
    km.fit(data)
    pred = km.predict(data)
    centers = km.cluster_centers_

    print_scores(labels, km.labels_)

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

cluster_plot(labels, lsi.fit_transform(tfidf), "", "", "SVD Visualization")
cluster_plot(labels, nmf.fit_transform(tfidf), "", "", "NMF Visualization")

#part 4b
from sklearn.preprocessing import scale, FunctionTransformer
print "SVD with unit variance"
cluster_plot(labels, scale(lsi.fit_transform(tfidf)), "", "", "SVD with unit variance visualization")
print "NMF with unit variance"
cluster_plot(labels, scale(nmf.fit_transform(tfidf)), "", "", "NMF with unit variance visualization")


log_tf = FunctionTransformer(func=np.log, inverse_func=np.exp)
print "NMF with non-linear (log) transform"
cluster_plot(labels, log_tf.fit_transform(nmf.fit_transform(tfidf) + 1), "", "", "NMF with logarithm transform visualization")

print "NMF with scale then log transform"
cluster_plot(labels, log_tf.fit_transform(scale(nmf.fit_transform(tfidf)) + 1), "", "", "NMF with unit variance then logarithm transform visualization")

print "NMF with log transform then scale"
cluster_plot(labels, log_tf.fit_transform(scale(nmf.fit_transform(tfidf)) + 1), "", "", "NMF with logarithm transform then unit variance visualization")
