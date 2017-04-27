# Created by JJW Apr 25 2016
# This will perform a PCA analysis on the data
# in order to try to discover any groupings that can be used
# to classify the data
# for COS 424 final project
#
# I was helped by http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import general_functions as gf
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer

import numpy as np


# Some parameters for this run
n_components = 5
imputation_strategey = 'median'

data, category = gf.read_in_data()

# Convert inf to nan for imputation
for i in range(len(data)):
    for j in range(len(data[i])):
        if np.isinf(data[i][j]):
            data[i][j] = float('nan')

# Impute
imputation = Imputer(missing_values='NaN', strategy=imputation_strategey)
data = imputation.fit_transform(data)

pca = PCA(n_components=n_components, copy=True)

pca.fit(data)

print(pca.explained_variance_ratio_) 
