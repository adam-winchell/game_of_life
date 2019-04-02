import os
import cellular_automaton as ca

if __name__ == "__main__":

	files = []
	for file in os.listdir("seeds"):
		if 'oscillator' in file:
			files.append(file)
			

	fitnesses = []
	for filename in files:
		fitness_values = getattr(ca, 'main')(time_steps=16, gif_on=False, seed=filename, individual_fitness=True)
		print(fitness_values)
		quit()
		#run the oscillator and get it's fitness for each f_i and store this result in fitnesses
		pass

	#find weights w1, w2, w3 such that for each row in fitnesses the total fitness ~1