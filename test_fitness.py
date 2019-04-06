import os
import cellular_automaton as ca

if __name__ == "__main__":

	files = []
	for file in os.listdir("seeds/"):
		#added temporary conditions to isolate period 15 oscillators
		if 'oscillator' in file and ('Penta' in file or '112P15' in file):
			files.append(file)
			

	fitnesses = []
	for filename in files:
		splt = filename.split('_')
		print(filename) 

		print(int(splt[2])+1," timesteps")
		fitness_values = ca.main(time_steps=(int(splt[2])+1), filename=filename,gif_on=False, seed=filename, individual_fitness=True)
		print(int(splt[2])*2," timesteps")
		ca.main(time_steps=(int(splt[2])*2), filename=filename,gif_on=True, seed=filename, individual_fitness=True)
		#print(fitness_values)
		print()

		#run the oscillator and get it's fitness for each f_i and store this result in fitnesses
		pass

	#find weights w1, w2, w3 such that for each row in fitnesses the total fitness ~1