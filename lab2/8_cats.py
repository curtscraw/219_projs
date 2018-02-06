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
from sklearn.metrics import confusion_matrix

print "Clustering sparse tfidf with kmeans, k=2"

#returns [homogeneity, completeness, v_measure, rand, mutualr_info] for plotting 
def print_scores(labels, predicted):
    print "Contingency: "
    print str(confusion_matrix(labels, predicted))

    ret = [] 
    ret.append(homogeneity_score(labels, predicted))
    ret.append(completeness_score(labels, predicted))
    ret.append(v_measure_score(labels, predicted))
    ret.append(adjusted_rand_score(labels, predicted))
    ret.append(adjusted_mutual_info_score(labels, predicted))

    print "Homogeneity: " + str(homogeneity_score(labels, predicted))
    print "completeness: " + str(completeness_score(labels, predicted))
    print "V-measure: " + str(v_measure_score(labels, predicted))
    print "RAND score: " + str(adjusted_rand_score(labels, predicted))
    print "Mutual Info: " + str(adjusted_mutual_info_score(labels, predicted))
    
    return ret

#returns [homogeneity, completeness, v_measure, rand, mutualr_info] for plotting 
def km_score(labels, data):
    km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
    km.fit(data)

    return print_scores(labels, km.labels_)

#don't care to save the returned values
temp = km_score(labels, tfidf)

#part 3
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD

#run truncated svd with target of 1000, get variance retained by each of fist 1000 principal components
lsi = TruncatedSVD(n_components=1000, random_state=0)
lsi.fit(tfidf)
lsi_var = lsi.explained_variance_ratio_.cumsum().tolist()

#3ai, only need to do SVD! thank goodness!
rvals = range(1, 1001)
plt.plot(rvals, lsi_var, label="SVD")
plt.xlabel('r value')
plt.ylabel('Percent of variance retained')
plt.title('Variance retained vs r value for Truncated SVD')
plt.show()

#truncated svd version
nmf_scores = []
lsi_scores = []
key_r = [1, 2, 3, 5, 10, 20, 50, 100, 300]
for i in key_r:
    nmf = NMF(n_components=i, init='random', random_state=0)
    lsi = TruncatedSVD(n_components=i, random_state=0)

    print "NMF with r=" + str(i)
    nmf_scores.append(km_score(labels, nmf.fit_transform(tfidf)))
    print "SVD with r=" + str(i)
    lsi_scores.append(km_score(labels, lsi.fit_transform(tfidf)))

#plots for part 3aii
#homogeneity
plt.plot(key_r, [pt[0] for pt in lsi_scores], label="SVD")
plt.plot(key_r, [pt[0] for pt in nmf_scores], label="NMF")
plt.xlabel('r value')
plt.ylabel('Homogeneity Score')
plt.title('Homogeneity Score relative to r value')
plt.legend(loc="lower right")
plt.show()

#completeness
plt.plot(key_r, [pt[1] for pt in lsi_scores], label="SVD")
plt.plot(key_r, [pt[1] for pt in nmf_scores], label="NMF")
plt.xlabel('r value')
plt.ylabel('Completeness Score')
plt.title('Completeness Score relative to r value')
plt.legend(loc="lower right")
plt.show()

#v_measure
plt.plot(key_r, [pt[2] for pt in lsi_scores], label="SVD")
plt.plot(key_r, [pt[2] for pt in nmf_scores], label="NMF")
plt.xlabel('r value')
plt.ylabel('V-measure Score')
plt.title('V-measure Score relative to r value')
plt.legend(loc="lower right")
plt.show()

#RAND-index
plt.plot(key_r, [pt[3] for pt in lsi_scores], label="SVD")
plt.plot(key_r, [pt[3] for pt in nmf_scores], label="NMF")
plt.xlabel('r value')
plt.ylabel('RAND Score')
plt.title('Adjusted RAND relative to r value')
plt.legend(loc="lower right")
plt.show()

#mutual info
plt.plot(key_r, [pt[4] for pt in lsi_scores], label="SVD")
plt.plot(key_r, [pt[4] for pt in nmf_scores], label="NMF")
plt.xlabel('r value')
plt.ylabel('Mutual Info Score')
plt.title('Adjusted Mutual Info Score relative to r value')
plt.legend(loc="lower right")
plt.show()
 
#part 4
#TODO
