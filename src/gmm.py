# Created by JJW Apr 28 2016
# This will perform an LDA analysis on the data
# in order to try to discover any groupings that can be used
# to classify the data
# for COS 424 final project
#
# I was helped by http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import general_functions as gf
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import Imputer
import soft_impute
#from sklearn.preprocessing import scale

import numpy as np


def softImpute(data, nCompSoft=30, **kargs):
        imputeFnc = soft_impute.SoftImpute(J=nCompSoft, lambda_=0.0)
        fit = imputeFnc.fit(data)
        imputeFnc.predict(data, copyto=True)
        return data

# Some parameters for this run
n_topics = 2

data, category = gf.read_in_data()

# Convert inf to nan for imputation
for i in range(len(data)):
    for j in range(len(data[i])):
        if np.isinf(data[i][j]):
            data[i][j] = float('nan')
data = softImpute(np.array(data))

test_data = np.concatenate((data[:150], data[-400:]))
test_category = np.concatenate((category[:150], category[-400:]))
training_data = data[150:-400]
training_category = category[150:-400]

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
print "---"
#print len([i[0] for i in predictions_posterior[:874] if i[0] >0.75])
#print len([i[1] for i in predictions_posterior[:874] if i[1] >0.75])
#print len([i[2] for i in predictions_posterior[:874] if i[2] >0.75])
print "----"
print "-----"
print len([i[0] for i in predictions_posterior[874:] if i[0] >0.9])
print len([i[1] for i in predictions_posterior[874:] if i[1] >0.9])
print len([i[2] for i in predictions_posterior[874:] if i[2] >0.9])
print len([i[3] for i in predictions_posterior[874:] if i[3] >0.9])
#print len([i[4] for i in predictions_posterior[874:] if i[4] >0.9])
print "----"
#print len([i[0] for i in predictions_posterior[874:] if i[0] >0.75])
#print len([i[1] for i in predictions_posterior[874:] if i[1] >0.75])
#print len([i[2] for i in predictions_posterior[874:] if i[2] >0.75])

