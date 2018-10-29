import numpy as np
import math, sys

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
			
			yield train_index, test_index




class randomSplit():
	def __init__(self, num_splits, train_percent):
		self.num_splits = num_splits
		self.train_percent = train_percent

	def generateIndex(self, y, percent):
		index_all = np.arange(y.shape[0])
		labels = np.unique(y)

		train_index = np.array([], dtype=int)

		for i in range(labels.size):
			label_index = index_all[np.where(y==labels[i])]
			temp = np.random.choice(label_index, math.ceil(label_index.size * percent), replace=False)
			train_index = np.concatenate((train_index, temp))

		test_index = np.setdiff1d(index_all, train_index, assume_unique=True)

		return train_index, test_index

	def split(self, y):

		for i in range(self.num_splits):

			print('The',i+1,'split of 10 random 80-20 train-test splits')
			print()

			train, test_index = self.generateIndex(y, 0.8)

			y_train = y[train]

			for p in range(len(self.train_percent)):
				print('Using',self.train_percent[p],'%'+'of traning data')
				print()

				train_index,_ = self.generateIndex(y_train, self.train_percent[p]/100)

				#print('Training data size:', train_index.size)
				#print('Test data size:', test_index.size)
				#print()
				yield train_index, test_index, i, p