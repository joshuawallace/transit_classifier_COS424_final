# Created by JJW Apr 25 2016
# This will perform a k-means clustering analysis on the data
# in order to try to discover any groupings that can be used
# to classify the data
# for COS 424 final project
#
# I was helped by http://scikit-learn.org/stable/tutorial/statistical_inference/unsupervised_learning.html#clustering-grouping-observations-together

import general_functions as gf
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import Imputer
import soft_impute

import numpy as np


# Some parameters for this run
number_of_clusters = 7
num_cross_validation_folds = 10
imputation_strategey = 'median'


def softImpute(data, nCompSoft=30, **kargs):
        imputeFnc = soft_impute.SoftImpute(J=nCompSoft, lambda_=0.0)
        fit = imputeFnc.fit(data)
        imputeFnc.predict(data, copyto=True)
        return data


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


clustering = KMeans(n_clusters=number_of_clusters, n_jobs=-2, n_init=12, init='random')

# Create a KFold instance
cross_val_fold = KFold(n_splits=num_cross_validation_folds, shuffle=True)

for training_indices, testing_indices in cross_val_fold.split(data):
    clustering.fit([data[i] for i in training_indices])# [category[i] for i in training_indices])
    #print clustering.labels_[:150]
    #print clustering.labels_[-150:]
    print len([i for i in range(len(clustering.labels_)) if clustering.labels_[i] == 0])
    print len([i for i in range(len(clustering.labels_)) if clustering.labels_[i] == 1])
    print len([i for i in range(len(clustering.labels_)) if clustering.labels_[i] == 2])
    print len([i for i in range(len(clustering.labels_)) if clustering.labels_[i] == 3])
    print len([i for i in range(len(clustering.labels_)) if clustering.labels_[i] == 4])
    print "------"
    list_ = [ ]
    for i in range(number_of_clusters):
        list_.append([])
    a = clustering.predict([data[i] for i in testing_indices])
    for i in range(len(a)):
        list_[ a[i]].append(category[testing_indices[i]])
    for i in range(len(list_)):
        print len([j for j in list_[i] if j == 1])

    print ""
    print ""
    print ""
    print ""







