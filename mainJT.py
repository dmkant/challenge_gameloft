import pandas as pd
from gensim.models import Word2Vec
import nltk
from gensim.models import KeyedVectors

from nltk.cluster import KMeansClusterer
import numpy as np

from sklearn import cluster
from sklearn import metrics

import spacy

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


DC_US_GP = pd.read_csv(
    r"c:\Users\jason\OneDrive\Bureau\challenge_gameloft\data\reviews_DC_US_GP.csv", sep="\t")

# print(type(D[0, 0]))


copy_DC_US_GP = [i for i in DC_US_GP["Content"]]


df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),

                   columns=['a', 'b', 'c'])

ve = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(np.mean(ve, 0))

"""
print(df2)
print(df2["a"])
print(df2.loc[1, "a"])
print(df2.iloc[1])
print(df2.loc[df2["a"] >= 4, :])
"""


nlp = spacy.load("en_core_web_md")
test = [doc for doc in nlp.pipe(
    DC_US_GP["Content"].iloc[:10].astype(str), disable=['parser', 'ner'])]  # liste des 10 premiers commentaires


def list_vectb(phrase):
    l = []
    for m in phrase:
        l.append(m.vector)
    return l

# list_vect est une liste de vecteur correspondant au vecteur des mots d'une phrase


def phrase_to_vec(list_vect):  # renvoie le vecteur d'une phrase
    res = np.zeros(300)
    n = 0
    for v in list_vect:
        if sum(v) != 0:
            res += v
            n += 1
    return res/n


def list_vect(tes):
    l = []
    for p in tes:
        # print(p)
        l.append(phrase_to_vec(list_vectb(p)))
    return l


print(test[0])
print(test[0][0].text)
X = list_vect(test)

kclusterer = KMeansClusterer(
    2, distance=nltk.cluster.util.cosine_distance, repeats=25)
clusters = kclusterer.cluster(X, assign_clusters=True)
print(clusters)

# for index, phrase in enumerate(phrases):
#     print (str(clusters[index]) + ":" + str(phrase))

"""
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(X)


model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)

Y = model.fit_transform(X)
plt.scatter(Y[:, 0], Y[:, 1], c=clusters, s=290, alpha=.5)

plt.show()

# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
"""

# print(test[0])
# print(test[0][6])
# print(sum(test[0][7].vector))
# print(test[0][6].lemma_)
