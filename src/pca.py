# Created by JJW Apr 25 2016
# This will perform a PCA analysis on the data
# in order to try to discover any structure that can be used
# to classify the data
# for COS 424 final project
#
# I was helped by http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import general_functions as gf
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import svm
import soft_impute

import numpy as np


def softImpute(data, nCompSoft=30, **kargs):
        imputeFnc = soft_impute.SoftImpute(J=nCompSoft, lambda_=0.0)
        fit = imputeFnc.fit(data)
        imputeFnc.predict(data, copyto=True)
        return data


# Some parameters for this run
n_components = 60

data, category = gf.read_in_data()

# Convert inf to nan for imputation
for i in range(len(data)):
    for j in range(len(data[i])):
        if np.isinf(data[i][j]):
            data[i][j] = float('nan')
data = softImpute(np.array(data))

# Impute
#imputation = Imputer(missing_values='NaN', strategy=imputation_strategey)
#data = imputation.fit_transform(data)
data = scale(data)

test_data = np.concatenate((data[:150], data[-400:]))
test_category = np.concatenate((category[:150], category[-400:]))
training_data = data[150:-400]
training_category = category[150:-400]

pca = PCA(n_components=n_components, copy=True)

training_data = pca.fit_transform(training_data)

print training_data

positive = training_data[:874]
negative = training_data[874:]
fig = plt.figure(figsize=(5,2.2))
ax = fig.add_subplot(111)
ax.scatter([i[0] for i in negative], [i[1] for i in negative],s=4, color='blue', alpha= .022)
ax.scatter([i[0] for i in positive], [i[1] for i in positive],s=4, color='green', alpha= .19)
ax.set_xlabel("First PC")
ax.set_ylabel("Second PC")
ax.set_xlim(-15,15)
ax.set_ylim(-15,15)
fig.tight_layout()
fig.savefig("pca.png")

print(pca.explained_variance_ratio_)

print np.var([i[0] for i in negative])
print np.var([i[1] for i in negative])

print np.var([i[0] for i in positive])
print np.var([i[1] for i in positive])

print "----"

clustering = KMeans(n_clusters=4, n_jobs=-2, n_init=12, init='random')
clustering.fit(training_data)

test_data = pca.transform(test_data)
a = clustering.predict(test_data)

scores = gf.precision_recall_etc(a, test_category)
print scores

"""
a_positive = a[:150]
a_negative = a[150:]

print len([i for i in a_positive if i==0])
print len([i for i in a_positive if i==1])
print len([i for i in a_positive if i==2])
print len([i for i in a_positive if i==3])
#print len([i for i in a_positive if i==4])
print "-"
print len([i for i in a_negative if i==0])
print len([i for i in a_negative if i==1])
print len([i for i in a_negative if i==2])
print len([i for i in a_negative if i==3])
#print len([i for i in a_positive if i==4])"""

clf = svm.SVC()
clf.fit(training_data, training_category)

#test_data = pca.transform(test_data)
predictions = clf.predict(test_data)

scores = gf.precision_recall_etc(predictions, test_category)
print scores


print "----"
print "--------"

gmm = GaussianMixture(n_components=4, n_init=5, init_params='random')
gmm.fit(training_data)
predictions = gmm.predict(training_data)
predictions_posterior = gmm.predict_proba(training_data)

print predictions
print predictions_posterior

print len([i[0] for i in predictions_posterior[:874] if i[0] >0.9])
print len([i[1] for i in predictions_posterior[:874] if i[1] >0.9])
print len([i[2] for i in predictions_posterior[:874] if i[2] >0.9])
print len([i[3] for i in predictions_posterior[:874] if i[3] >0.9])
#print len([i[4] for i in predictions_posterior[:874] if i[4] >0.9])
print "----"
print "-----"
print len([i[0] for i in predictions_posterior[874:] if i[0] >0.9])
print len([i[1] for i in predictions_posterior[874:] if i[1] >0.9])
print len([i[2] for i in predictions_posterior[874:] if i[2] >0.9])
print len([i[3] for i in predictions_posterior[874:] if i[3] >0.9])
#print len([i[4] for i in predictions_posterior[874:] if i[4] >0.9])
