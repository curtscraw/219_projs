"""
Curtis Crawford, 805024638
Abdullah-Al-Zubaer Imran, 804733867

EE219, Winter 2018
Project 1
"""

import numpy as np
import sklearn as skl
from sklearn.datasets import fetch_20newsgroups
import sklearn.datasets as skd
from sklearn.feature_extraction import text
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import sys

if len(sys.argv) != 3 or (sys.argv[2] != "lsi" and sys.argv[2] != "nmf"):
    print "Script must be called with a value for min_dif as the 1st argument, and either nmf or lsi as the second argument"
    exit()

this_df = int(sys.argv[1])
reuc = sys.argv[2]


print "This run will use min_df=" + str(this_df)

print "numpy version: " + np.__version__
print "sklearn version: " + skl.__version__
print "matplotlib version: " + mpl.__version__
print "nltk version: " + nltk.__version__

fullset = fetch_20newsgroups(shuffle=True, random_state=42);
trainset = fetch_20newsgroups(subset='train', shuffle=True, random_state=42);
testset = fetch_20newsgroups(subset='test', shuffle=True, random_state=42);

cset = [' ']*20

#convert C into a list of 20 total values
for i in range(0, len(fullset.target)):
    target = fullset.target[i]
    data = fullset.data[i]
    cur = cset[target]
    cset[target] = cur + " " + data

#include the stemmer in the countvectorizer class
stemmer = nltk.stem.porter.PorterStemmer()
class stemCV(text.CountVectorizer):
    def build_analyzer(self):
        analyzer = super(stemCV, self).build_analyzer()
        return lambda doc: ([stemmer.stem(t) for t in analyzer(doc)])

#vectorizer = text.CountVectorizer(min_df=this_df, stop_words='english')
vectorizer = stemCV(min_df=this_df, stop_words='english')

#tokenize, vectorize the stemmed data list
X = vectorizer.fit_transform(cset)

#find TFxICF
tfidf_transformer = text.TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)

#print out 10 largest TFxICF from the four categories
for cat in ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']:
    idx = fullset.target_names.index(cat)
    print cat + ":"
    row = X_tfidf.toarray()[idx].tolist()

    for i in range(10):
        idx_mc = row.index(max(row))
        most_common = vectorizer.get_feature_names()[idx_mc]
        print "\t" + str(i) + "th most common item is: " + str(most_common)
        row[idx_mc] = min(row)
