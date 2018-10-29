import numpy as np
import sys, os, math, random

from cvxopt import matrix, solvers

NUM_CROSSVAL=10
MIN_MULTIPLIER_THRESHOLD=1e-8


class KFold():
	def __init__(self, num_crossval, shuffle = True):
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
		
		# Support vectors have non zero lagrange multipliers
		index = alphas > 1e-8
		self.a = alphas[index]
		self.sv_y = self.y[index]
		self.sv = self.X[np.where(index), :]
		self.sup_num = len(self.a)
		print("%d support vectors out of %d points" % (self.sup_num, self.num))

		# get bias
		for n in range(len(self.a)):
			self.bias += self.sv_y[n]
			self.bias -= np.sum(self.a * self.sv_y)
		self.bias /= len(self.a)
		print("Bias:",self.bias)


		self.w = np.sum(alphas * self.y * self.X, axis = 0)

		return self

	def predict(self, X_test):
		print()
		print("Test data size: ", X_test.shape)
		y_pre = np.array([], dtype=int)
		for x in X_test:
			pre = np.sign(np.dot(self.w.T,x)+self.bias)
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

	def getSupNum(self):
		return self.sup_num

	def getWeight(self):
		return self.w


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
	margin = np.zeros((NUM_CROSSVAL, 1))
	support_num = np.zeros((NUM_CROSSVAL, 1))

	rs = KFold(num_crossval=NUM_CROSSVAL)
	for train_index, test_index, i in rs.split(y):
		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]

		model = SVM(C).fit(X_train, y_train)
		error[i] = model.score(X_test, y_test)
		support_num[i] = model.getSupNum()
		margin[i] = 1/np.linalg.norm(model.getWeight())

	print('Error mean:',np.mean(error))
	print('Error std:',np.std(error))
	print('Number of support vectors mean:',np.mean(support_num))
	print('Number of support vectors std:',np.std(support_num))
	print('Margin mean:',np.mean(margin))
	print('Margin std:',np.std(margin))

	return np.mean(error), np.std(error), np.mean(support_num), np.std(support_num), np.mean(margin), np.std(margin)


def main():
	filename = sys.argv[1]

	f = open('myDualSVM_result.csv','w')
	f.write('c,error_mean,error_std,sv_num_mean,sv_num_std,margin_mean,margin_std\n')
	for c in [0.01, 0.1, 1, 10, 100]:
		error_mean,error_std,sv_num_mean,sv_num_std,margin_mean,margin_std=myDualSVM(filename, c)
		output =str(c)+','+str(error_mean)+','+str(error_std)+','+str(sv_num_mean)+','+str(sv_num_std)+','+str(margin_mean)+','+str(margin_std)+'\n'
		f.write(output)
		#f.write('%.3f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f' %(c,error_mean,error_std,sv_num_mean,sv_num_std,margin_mean,margin_std))

	f.close()


if __name__ == '__main__':
	main()