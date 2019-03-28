import numpy as np
import pickle
from collections import defaultdict
from scipy.spatial.distance import pdist

def ff1(data):
	score = 0
	add1 = np.vectorize(lambda x: x+1 if x > 0 else 0)
	for d in range(1,len(data)):
		data[d] = add1(data[d])
		for i in range(data[d].shape[0]):
			for j in range(data[d].shape[0]):
				if data[d][i,j] > 0:
					score += data[d-1][i,j]
					data[d][i,j] = 1
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
	return score


# 	return score

# def ff3(data):



def main():
	with open('frames.p','rb') as pFile:
		data = pickle.load(pFile)
	print(ff3(data))


if __name__ == '__main__':
	main()



