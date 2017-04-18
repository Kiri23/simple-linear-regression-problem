# Example of Standalone Simple Linear Regression
from math import sqrt

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Evaluate regression algorithm on training dataset
def evaluate_algorithm(dataset, algorithm):
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(dataset, test_set)
	print(predicted)
	actual = [row[-1] for row in dataset]
	rmse = rmse_metric(actual, predicted)
	return rmse

# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions


# el dataset sin los numeros alto tiene un error de 75.118 , 
# los output sin los numeros fueron [10,8.39],[30,24,4],[50,40.4],[68,48.4],[200,160.4],[2000000,1600000.4],[999,799.6]
# con este dataset = [1, 1], [2, 3], [4, 3], [3, 2], [5, 5]
# ,[10,10],[30,30],[50,50],[60,60],[200,200],[200000,2000000],[999,999]
# [[1, 1], [2, 2], [4, 4], [3, 3], [5, 5]] - con este dataset haces las prediciones perfec
# Test simple linear regression
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5],[1000000,1000010],[900,890]]
# el algoritmo predice el ouput 
train = [ [10], [30], [50], [60], [200],[999] ]
prediction = simple_linear_regression(dataset,train);
print(prediction);
# [8.39,24.4,40.4,48.4,160.4,16000000.4,799.6]
rmse1 = rmse_metric([10,30,50,60,200,2000000,999],prediction)
print(rmse1)
# rmse1 = evaluate_algorithm(dataset,train, simple_linear_regression)
# print('RMSE: %.3f' % (rmse1))

rmse = evaluate_algorithm(dataset, simple_linear_regression)
print('RMSE: %.3f' % (rmse))