import numpy as np
import sys, os, math, random

from cvxopt import matrix, solvers

NUM_CROSSVAL=10
MIN_MULTIPLIER_THRESHOLD=1e-8


class KFold():
	def __init__(self, num_crossval, shuffle = False):
		self.num_crossval = num_crossval
		self.shuffle = shuffle

	def split(self, y):
		index_all = np.arange(y.shape[0])
		if (self.shuffle):
			np.random.shuffle(index_all)

		test_index_num = math.floor(index_all.size/self.num_crossval)

		reminder = index_all.size - test_index_num*self.num_crossval
		
		for i in range(self.num_crossval):

			print('The',i+1,'fold of K-Fold cross-validation')
			print()

			if(i < reminder):
				test_index = index_all[i*test_index_num+i:(i+1)*test_index_num+i+1]
			else:
				test_index = index_all[i*test_index_num+reminder:(i+1)*test_index_num+reminder]

			train_index = np.setdiff1d(index_all, test_index, assume_unique=True)
			
			yield train_index, test_index, i


class SVM():
	def __init__(self, C):
		self.C = C
		self.bias = 0
	
	def fit(self, X_train, y_train):
		self.X = X_train
		self.y = y_train
		self.labels = np.unique(y_train)
		self.label_num = len(self.labels)		# number of unique labels
		self.num = X_train.shape[0]				# number of instances
		self.feature_num = X_train.shape[1]		# number of features

		print("Class number: ", self.label_num)
		print("Training data size: ", self.X.shape)
		print("C:", self.C)
		print()
		
		
		K = self.y.reshape(-1,1)*self.X
		K = np.dot(K, K.T)

		P = matrix(K, K.shape, 'd')
		q = matrix(-np.ones((self.num, 1)))
		G = matrix(np.concatenate((np.eye(self.num), -np.eye(self.num))))
		h = matrix(np.concatenate((np.ones((self.num))*self.C, np.zeros(self.num))))

		y = self.y.reshape(1, -1)
		A = matrix(y, y.shape, 'd')
		b = matrix(np.zeros(1))

		sol = solvers.qp(P, q, G, h, A, b)
		alphas = np.array(sol['x'])
		'''
		# get bias
		cond = (alphas > 1e-4).reshape(-1)
		b = y[cond] - np.dot(x[cond], w)
		bias = b[0]
		'''

		self.w = np.sum(alphas * self.y * self.X, axis = 0)

		return self

	def predict(self, X_test):
		print("Test data size: ", X_test.shape)
		y_pre = np.array([], dtype=int)
		for x in X_test:
			pre = np.sign(np.dot(self.w.T,x))
			y_pre = np.append(y_pre, pre)

		return y_pre

	def score(self, X_test, y_test):
		y_pre = self.predict(X_test)

		result = np.array([], dtype=int)
		num = X_test.shape[0]
		for i in range(num):
			result = np.append(result, y_pre[i] == y_test[i])

		error = 1 - np.sum(result)/num

		print("Test error rate:",error)
		print()

		return error


def myDualSVM(filename, C):

	if not os.path.isfile(filename):
		sys.exit(
			"ERROR: File does not exist"
		)

	data = np.genfromtxt(filename, delimiter=',', dtype=int)

	X = data[:,1:]
	y = data[:,0].reshape(-1,1)

	labels = np.unique(y)
	for i in range(y.shape[0]):
		if y[i] == labels[0]:
			y[i] = -1
		elif y[i] == labels[1]:
			y[i] = 1
	
	error = np.ones((NUM_CROSSVAL, 1))

	rs = KFold(num_crossval=NUM_CROSSVAL)
	for train_index, test_index, i in rs.split(y):
		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]

		model = SVM(C).fit(X_train, y_train)
		error[i] = model.score(X_test, y_test)

	error_mean = np.mean(error)
	error_std = np.std(error)
	print('Error rate for each fold:')
	print(error)
	print('Error mean:',error_mean)
	print('Error std:',error_mean)

	return error_mean


def main():
	filename = sys.argv[1]
	C = float(sys.argv[2])

	myDualSVM(filename, C)


if __name__ == '__main__':
	main()