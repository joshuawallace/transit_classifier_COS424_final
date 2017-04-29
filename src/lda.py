# Created by JJW Apr 28 2016
# This will perform an LDA analysis on the data
# in order to try to discover any groupings that can be used
# to classify the data
# for COS 424 final project
#
# I was helped by http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import general_functions as gf
from sklearn.decomposition import LatentDirichletAllocation as LDA
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
n_topics = 5

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

#Make everything not negative
minimum_values = np.amin(data,axis=0)
minimum_values = [i if i<0 else 0. for i in minimum_values]
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] += -1.*minimum_values[j]

test_data = np.concatenate((data[:150], data[-400:]))
test_category = np.concatenate((category[:150], category[-400:]))
training_data = data[150:-400]
training_category = category[150:-400]

lda = LDA(n_topics=n_topics, doc_topic_prior=0.0001,
          topic_word_prior=0.0001,
          learning_method='batch', n_jobs=3)

training_data = lda.fit_transform(training_data)

print len([i[0] for i in training_data[:874] if i[0] >0.8])
print len([i[1] for i in training_data[:874] if i[1] >0.8])
print len([i[2] for i in training_data[:874] if i[2] >0.8])
print len([i[3] for i in training_data[:874] if i[3] >0.8])
print len([i[4] for i in training_data[:874] if i[4] >0.8])
print "----"
print len([i[0] for i in training_data[874:] if i[0] >0.8])
print len([i[1] for i in training_data[874:] if i[1] >0.8])
print len([i[2] for i in training_data[874:] if i[2] >0.8])
print len([i[3] for i in training_data[874:] if i[3] >0.8])
print len([i[4] for i in training_data[874:] if i[4] >0.8])


