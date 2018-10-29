import numpy as np
import sys, os, math, time, random
import matplotlib.pyplot as plt

class SVM():
	def __init__(self, k, _lambda = 100):
		self.batch_size = k
		self._lambda = _lambda
		self.obj_value = np.array([])
	
	def getObjValue(self):
		return self.obj_value

	def fit(self, X_train, y_train):
		self.X = X_train
		self.y = y_train
		self.labels = np.unique(y_train)
		self.label_num = len(self.labels)		# number of unique labels
		self.num = X_train.shape[0]				# number of instances
		self.feature_num = X_train.shape[1]		# number of features
		
		self.w = np.zeros((self.feature_num,1))
		self.itr_num = 100*self.num

		print("Class number: ", self.label_num)
		print("Training data size: ", self.X.shape)
		print("Batch size:", self.batch_size)
		print()

		ktot = self.itr_num
		self.w.fill(math.sqrt(1/self._lambda))


		for i in range(int(ktot/self.batch_size)):
			print('Number of iteration:', i)
			if self.batch_size != 1:
				index = self.batchIndex(self.y, self.batch_size)
				X_batch = self.X[index]
				y_batch = self.y[index]

			else:
				index = random.sample(range(self.num), 1)
				X_batch = self.X[index]
				y_batch = self.y[index]

			eta = 1/(self._lambda*(i+1))

			y_batch = y_batch.reshape(-1,1)
			loss = 1 - y_batch * np.dot(X_batch, self.w)
			#print(loss)

			index = (loss > 0)
			index = index.reshape((-1,))


			X_plus = X_batch[index]
			y_plus = y_batch[index]

			w_half = (1 - eta*self._lambda)*self.w + (eta*np.sum(y_plus * X_plus, axis=0)/self.batch_size).reshape(-1,1)

			#print(w_half)
			w_new = min(1, math.sqrt(1/self._lambda)/np.linalg.norm(w_half))*w_half
			

			obj = self._lambda/2 * np.dot(w_new.T, w_new) + np.mean(loss[loss>0])
			self.obj_value = np.append(self.obj_value, obj)
			
			if np.sum((w_new - self.w)**2) < 1e-5:
				print("Converge")
				print("Number of iteration:",i)
				self.itr_num = i
				break
			else:
				self.w = w_new
			
			#self.w = w_new
		
		return self

	def batchIndex(self, y, batch_size):
		percent = batch_size/y.size
		index_all = np.arange(y.shape[0])
		labels = np.unique(y)

		index = np.array([], dtype=int)
		number = np.array([], dtype=int)

		for i in range(labels.size):
			label_index = index_all[np.where(y==labels[i])]
			number = np.append(number, math.floor(label_index.size * percent))

		while np.sum(number) < batch_size:
			number[np.argmin(number)] += 1

		for i in range(number.size):
			label_index = index_all[np.where(y==labels[i])]
			temp = np.random.choice(label_index, number[i], replace=False)
			index = np.concatenate((index, temp))

		return index


def myPegasos(filename, k, numruns):

	if not os.path.isfile(filename):
		sys.exit(
			"ERROR: File does not exist"
		)

	data = np.genfromtxt(filename, delimiter=',', dtype=int)

	X = data[:,1:]
	y_raw = data[:,0]

	y = np.zeros(y_raw.shape, dtype=int)
	labels  = np.unique(y_raw)
	y[y_raw == labels[0]] = 1
	y[y_raw == labels[1]] = -1

	runtime = np.ones((numruns, 1))

	fig = plt.figure()

	for i in range(numruns):
		print('The',i+1,'run of total',numruns,'runs')
		start = time.time()
		model = SVM(k).fit(X, y)

		end = time.time()
		runtime[i] = end-start

		plt.plot(model.getObjValue(), ',-')

	plt.title('Batch size: '+str(k))
	plt.ylabel('Objective function value')
	plt.xlabel('Number of iteration')
	plt.legend()

	if not os.path.exists('img/'):
		os.makedirs('img/')
	path = 'img/'+str(k)+'batch_myPegasos.png'
	fig.savefig(path)

	runtime_mean = np.mean(runtime)
	runtime_std = np.std(runtime)
	print('Runtime for each fold:')
	print(runtime)
	print('Runtime mean:',runtime_mean)
	print('Runtime std:',runtime_std)

	return runtime_mean, runtime_std

def main():
	filename = sys.argv[1]
	numruns = int(sys.argv[2])
	np.random.seed(int(time.time()))

	f = open('myPegasos_result.csv','w')
	f.write('k,mean,std\n')

	for k in [1, 20, 200, 1000, 2000]:
		mean, std = myPegasos(filename, int(k), numruns)
		output = str(k)+','+str(mean)+','+str(std)+'\n'
		f.write(output)

	f.close()

if __name__ == '__main__':
	main()