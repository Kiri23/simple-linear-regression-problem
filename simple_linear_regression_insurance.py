# Example of Simple Linear Regression on the Swedish Insurance Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Split a dataset into a train and test set
def train_test_split(dataset, split):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
    # partio la data aqui, la test data se usa para predecir los ouput y comprarlo para ver si el algoritmo predice bien
	train, test = train_test_split(dataset, split)
	# test set sin el ouput 
	test_set = list()
	print("training data")
	print(dataset)
	print()
	print("test data")
	print(test)
	for row in test:
	    # hace una copia de los rows input y output
		row_copy = list(row)
		# pone el ultimo valor el output como none para que el algoritmo lop predija
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(train, test_set, *args)
	actual = [row[-1] for row in test]
	print("output de los test data ")
	print(actual)
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
	# del train data se saca los coeficiente 
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		# print("este es el verdadero valor ")
		# print(row)
		# print(row[0])
		# print("--------")
		predictions.append(yhat)
	print()
	print("output que se predijo del test data")	
	print(predictions)
	print()
	return predictions
	
	
	
	

# Simple linear regression on insurance dataset
seed(1)
# load and prepare data
filename = 'insurance.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
split = 0.6
data = [[10,50],[20,50],[30,50],[40,50],[30,60],[90,99],[50,70],[67,89],[96,78],[85,87]]
rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
print('porciento de error RMSE: %.3f' % (rmse))
