import numpy as np
import pickle
from collections import defaultdict
from copy import deepcopy
from scipy.spatial.distance import pdist


def ff1(data):
	score = 0
	result = deepcopy(data[0])

	add1 = np.vectorize(lambda x: x+1 if x > 0 else 0)
	for d in range(1,len(data)):


		for m in result:
			for n in m:
				if n > 0:
					print("%2d" % n,end=' ')
				else:
					print('__',end=' ')
			print()
		print()

		result = add1(result)


		for i in range(data[d].shape[0]):
			for j in range(data[d].shape[0]):
				if data[d][i,j] > 0:
					result[i,j] = 1

	print(np.max(result))		
	return np.max(result)/(len(data)-1)


#TODO run all the f_i in one fitness function becuase all of them need the result matrix

def ff2(data):
	score = 0
	add1 = np.vectorize(lambda x: x+1 if x > 0 else 0)
	for d in range(1,len(data)):
		data[d] = add1(data[d])
		lin = list(data[d].ravel())
		print(np.bincount(lin))
		#print(counts)
	return score

def ff3(data):
	score = 0
	for d in range(1,len(data)):
		dist_arr = []
		for i in range(data[d].shape[0]):
			for j in range(data[d].shape[0]):
				if data[d][i,j] > 0:
					dist_arr.append((i,j))
		dist = pdist(dist_arr)
		print(dist.sum()/len(dist))
		print(max(dist))
		print(min(dist))
		print()
		#print(pdist(dist_arr))
	return 1/score


def fitness(data, weight_vector=[0.33,0.33,0.33]):
	result = deepcopy(data[0])

	f1, f2 , f3 = -1, -1, -1

	add1 = np.vectorize(lambda x: x+1 if x > 0 else 0)
	for d in range(1,len(data)):
		result = add1(result)

		for i in range(data[d].shape[0]):
			for j in range(data[d].shape[0]):
				if data[d][i,j] > 0:
					result[i,j] = 1

		f1 = max(f1, np.max(result))

		indices = np.transpose(np.nonzero(result))


		bins = defaultdict(list)
		for index in indices:
			bins[result[index[0], index[1]]].append(index)

		max_oscillating_cells = (-1,-1)	#	(key, num oscillating cells)
		for key, vals in bins.items():
			if len(vals) > max_oscillating_cells[1]:
				max_oscillating_cells = (key, len(vals))


		if max_oscillating_cells[1] > f2:
			f2 = max_oscillating_cells[1]

			dist = pdist(bins[max_oscillating_cells[0]])
			f3 = 1 / (dist.sum()/max_oscillating_cells[1])


	#normalize the fitness values to range [0,1]
	f1 = f1 / (len(data) - 1)
	f2 = f2 / (result.shape[0]*result.shape[1])
	#f3 is implicity normalized
	print(f1)
	print(f2)
	print(f3)
	return np.dot(weight_vector, [f1,f2,f3])


def main():
	with open('frames.p','rb') as pFile:
		data = pickle.load(pFile)
	print(fitness(data))


if __name__ == '__main__':
	main()



