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

#calculate parameters for Gaussians
cov_M = numpy.cov(train_Z_M,rowvar=False)
cov_B = numpy.cov(train_Z_M,rowvar=False)
mu_M = train_Z_M.mean()
mu_B = train_Z_B.mean()

#esitmate P(X|Y) and P(Y) for each class
num_M = 0
num_B = 0
for diagnosis in train_Y_is_M:
    if diagnosis == True:
        num_M += 1
    else:
        num_B += 1
prob_M = num_M / len(train_Y) #P(Y=M)
prob_B = num_B / len(train_Y) #P(Y=B)
prob_X_given_M_train = mv_norm.pdf(train_Z,mu_M,cov_M) #P(X|Y=M)
prob_X_given_B_train = mv_norm.pdf(train_Z,mu_B,cov_B) #P(X|Y=B)
prob_X_given_M_test  = mv_norm.pdf(test_Z,mu_M,cov_M) #P(X|Y=M)
prob_X_given_B_test  = mv_norm.pdf(test_Z,mu_B,cov_B) #P(X|Y=B)
p_M_train = prob_X_given_M_train * prob_M
p_B_train = prob_X_given_B_train * prob_B
p_M_test  = prob_X_given_M_test * prob_M
p_B_test  = prob_X_given_B_test * prob_B

#make predictions based on estimated probabilities and calculate error
predictions_train = p_M_train > p_B_train
success_train = (predictions_train == train_Y_is_M).mean()
predictions_test = p_M_test > p_B_test
success_test = (predictions_test == test_Y_is_M).mean()
error_train = 1.0 - success_train
error_test = 1.0 - success_test

print('training error: {} || test errror: {}'.format(error_train,error_test))
