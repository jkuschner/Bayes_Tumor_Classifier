import pandas
from scipy.stats import multivariate_normal as mv_norm
import numpy.matlib
import numpy

train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')

train_X = train[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]
train_Y = train[['diagnosis']]
test_X = test[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]
test_Y = test[['diagnosis']]

#standardize data
mu = train_X.mean(axis = 0)
sigma = train_X.std(axis = 0)
train_Z = (train_X - mu) / sigma
test_Z = (test_X - mu) / sigma
train_Y_is_M = train_Y['diagnosis'] == 'M'
test_Y_is_M = test_Y['diagnosis'] == 'M'

#separate into 2 classes
is_benign = train_Y['diagnosis'] == 'B'
is_malignant = train_Y['diagnosis'] == 'M'
train_Z_B = train_Z[is_benign]
train_Z_M = train_Z[is_malignant]

#calculate covariance matrix
var_B = train_Z_B.std(axis = 0) ** 2
var_M = train_Z_M.std(axis = 0) ** 2
var_C = (len(train_Z_M) * var_M + len(train_Z_B) * var_B) / (len(train_Z_M) + len(train_Z_B))

#calculate decision boundary
mu_B = train_Z_B.mean(axis = 0)
mu_M = train_Z_M.mean(axis = 0)
w = (mu_M - mu_B) / var_C

#determine theta
distances_train = numpy.dot(train_Z,w)
max_success = 0.0
max_theta = -10
for theta in range(-9,10):
    predictions_train = distances_train > theta
    success_rate = (predictions_train == train_Y_is_M).mean()
    if success_rate > max_success:
        max_success = success_rate
        max_theta = theta

#use classifier on test set and determine error
distances_test = numpy.dot(test_Z,w)
predictions_test = distances_test > max_theta
test_rate = (predictions_test == test_Y_is_M).mean()
training_error = 1.0 - max_success
test_error = 1.0 - test_rate
print('training success rate: {} test success rate: {}'.format(max_success,test_rate))
print('training error: {} test error: {}'.format(training_error,test_error))
print('done')
