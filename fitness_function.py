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





def main():
	with open('frames.p','rb') as pFile:
		data = pickle.load(pFile)
	print(ff1(data))


if __name__ == '__main__':
	main()



