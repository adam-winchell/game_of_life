import os
import cellular_automaton as ca
import numpy as np
import matplotlib.pyplot as plt

def get_weights():
	#This function assumes that f0 get 0.5 weight and everything else shares weight from the remaining 0.5
	files = []
	for file in os.listdir("seeds/"):
		if 'oscillator' in file:
			files.append(file)

	fitnesses = []
	for filename in files:
		splt = filename.split('_')

		fitness_values = ca.main(time_steps=(int(splt[2]) * 2 + 2), filename=filename, gif_on=False, seed=filename)
		fitnesses.append(fitness_values)

	fitnesses = np.array(fitnesses)

	w0 = 0.5
	weights = [w0]  # w0 is 0.5 because f0 is a necessary condition for an oscillator so it will be priortized

	for i in range(1, fitnesses.shape[1]):
		top = np.max(fitnesses[:, i])
		weights.append((1 - w0) / ((fitnesses.shape[1] - 1) * top))

	return weights

if __name__ == "__main__":
	files = []
	for file in os.listdir("seeds/"):
		if 'oscillator' in file:
			files.append(file)


	fitnesses = []
	for filename in files:
		splt = filename.split('_')
		# print('**********')
		# print(filename)
		fitness_values = ca.main(time_steps=(int(splt[2])*2 + 2), filename=filename,gif_on=False, seed=filename)
		fitnesses.append(fitness_values)
		# print(fitness_values)
		# print(np.dot(fitness_values, weights))

	fitnesses = np.array(fitnesses)

	w0 = 0.5
	weights = [w0]	#w0 is 0.5 because f0 is a necessary condition for an oscillator so it will be priortized

	for i in range(1, fitnesses.shape[1]):
		top = np.max(fitnesses[:, i])
		weights.append((1-w0)/((fitnesses.shape[1]-1)*top))

	total_fitness = []
	for filename, fitness in zip(files, fitnesses):
		print('**********')
		print(filename)
		# print(np.dot(fitness, weights))
		print(fitness)
		total_fitness.append(np.dot(fitness, weights))

	plt.hist(total_fitness, bins=30)
	plt.xlabel('Total Fitness')
	plt.ylabel('Count')
	plt.show()








	# fig, ax = plt.subplots(nrows=2, ncols=3)
	#
	# ax[0,0].hist(fitnesses[:, 0], bins=20)
	# ax[0,0].set_title('F0')
	# ax[0, 1].hist(fitnesses[:, 1], bins=20)
	# ax[0, 1].set_title('F1')
	# ax[0, 2].hist(fitnesses[:, 2], bins=20)
	# ax[0, 2].set_title('F2')
	# ax[1, 0].hist(fitnesses[:, 3], bins=20)
	# ax[1, 0].set_title('F3')
	# ax[1, 1].hist(fitnesses[:, 4], bins=20)
	# ax[1, 1].set_title('F4')
	#
	# plt.show()