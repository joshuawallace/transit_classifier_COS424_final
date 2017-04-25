# Created by JJW Apr 24 2016
# This implements K-nearest neighbors
# for COS 424 final project
#
# Help from http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier


from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

import general_functions as general_f
import numpy as np


# Some parameters for this run
num_cross_validation_folds = 50
K_max = 90
K_min = 5
K_space = 5
n_neighbors_min = 3
n_neighbors_max = 12
n_neighbors_space = 1
imputation_strategey = 'median'

# Read in the data
data, category = general_f.read_in_data()

# Convert inf to nan for imputation
for i in range(len(data)):
    for j in range(len(data[i])):
        if np.isinf(data[i][j]):
            data[i][j] = float('nan')

# Impute
imputation = Imputer(missing_values='NaN', strategy=imputation_strategey)
data = imputation.fit_transform(data)

# Create a KFold instance
cross_val_fold = KFold(n_splits=num_cross_validation_folds, shuffle=True)

# The different values to use for K-best feature selection
k_values = range(K_min, K_max, K_space)

# The different number of neighbors to try
n_neighbors = range(n_neighbors_min, n_neighbors_max, n_neighbors_space)

# Loop over the different k_values for K-best feature selection
for val in k_values:
    feature_sel = SelectKBest(score_func=f_regression, k=val)
    # Loop over the numbers of neighbors
    for nn in n_neighbors:
        clf = KNC(n_neighbors=nn, n_jobs=3) # Define classifier
        pipeline = Pipeline([('select', feature_sel),
                            ('classifier', clf)])
        # Train and run over the different cross validation folds
        for training_indices, testing_indices in cross_val_fold.split(data):
            pipeline.fit([data[i] for i in training_indices], [category[i] for i in training_indices])
            prediction = pipeline.predict(data[testing_indices])

