import re
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
	t1 = '4b2o4b$5bo4b$4bo5b$3bob3o2b$3bobo2bob$2obo3bobo$2obo4bob$4b4o2b2$4b2o4b$4b2o!'
	t2 = '2o12b2o$bo12bob$bobo8bobob$2b2o8b2o2b2$5b6o5b2$2b2o8b2o2b$bobo8bobob$bo12bob$2o12b2o!'


	string = t2

	grid = []
	idx = 0
	temp = []
	row = []
	while True:

		if string[idx] == '!':
			for _ in range(len(grid[0]) - len(row)):
				row.append(0)
			grid.append(row)
			break

		if string[idx].isdigit():
			temp.append(string[idx])
			idx += 1
			
		elif string[idx].isalpha():
			if temp == []:
				num = 1
			else:
				num = int(''.join(temp))
			num_to_add = 1
			if string[idx] == 'b':
				num_to_add = 0 

			for _ in range(num):
				row.append(num_to_add)
			temp = []
			idx += 1


		elif string[idx] == '$':
			if temp == []:
				grid.append(row)
				row = []
			else:
				num = int(''.join(temp)) - 1	
				grid.append(row)
				for _ in range(num):
					grid.append([0 for _ in range(len(grid[0]))])
				temp = []
				row = []
			idx += 1

	grid = np.array(grid)
	plt.matshow(grid)
	plt.show()

		

	# grid = []
	# for r in t2.split('$'):
	# 	row = []
	# 	matches = re.findall('((\\d*o)|(\\d*b))',r)
	# 	for m in matches:
	# 		digits = re.findall('\\d*', m[0])
	# 		if digits[0] == '':
	# 			digits = 1
	# 		else:
	# 			digits = int(digits[0])

	# 		num_to_add = 1
	# 		if 'b' in m[0]:
	# 			num_to_add = 0

	# 		for _ in range(digits):
	# 			row.append(num_to_add)
	# 	if '!' in r and len(row) < len(grid[0]):
	# 		for _ in range(len(grid[0]) - len(row)):
	# 			row.append(0)

	# 	grid.append(row)

	# grid = np.array(grid)
	# plt.matshow(grid)
	# plt.show()
	

