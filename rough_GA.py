import numpy as np
import pickle
import matplotlib.pyplot as plt
import cellular_automaton as ca
from multiprocessing import Pool

#dumb board similarity function
def boardSim(board1,board2):
	total = 0
	count = 0
	for i in range(len(board1)):
		for j in range(len(board1)):
			total += board1[i,j] == board2[i,j]
			count += 1

	return total/count

#creates new generation from a list of GA objects
def generate_generation(agent_list):
	children = []
	for parent1 in agent_list:
		for parent2 in agent_list:
			children.append(parent1.crossover(parent2))
	return children


class GA:
	def __init__(self,ratio=0.2,fitness=0,board=[]):
		self.ratio = ratio #chance of life in random board
		self.fitness = fitness #fitness of this agent
		if len(board) == 0: #are we given a board or do we generate one?
			self.board = np.zeros(shape=(50,50))
			for i in range(50):
				for j in range(50):
					flip = np.random.uniform(0,1)
					if flip < ratio:
						self.board[i,j] = 1
		else:
			self.board = board

	def crossover(self,partner):
		#generate "chromosomes" from each parent GA
		#determine crossover point
		chromosome1 = np.ravel(self.board)
		chromosome2 = np.ravel(partner.board)
		split = np.random.randint(len(chromosome1))

		#combine parents chromosomes in child
		child = list(chromosome1[:split]) + list(chromosome2[split:])
		child = np.array(child)
		child = GA(board=np.reshape(child,(50,50)))

		#mutate if parents are identical
		if boardSim(self.board,partner.board) == 1:
			child.mutate()

		#chance of mutation if they're not
		else:
			flip = np.random.uniform(0,1)
			if flip < 0.1:
				child.mutate()


		return child

	def mutate(self):
		#mutation is just flipping one random element of the board
		self.board = np.ravel(self.board)
		flip = np.random.randint(len(self.board))
		self.board[flip] = 1 if self.board[flip] == 0 else 1
		self.board = np.reshape(self.board,(50,50))




'''changes to CA to make this work:
1) main takes a 'grid' parameter to pass each GA's board
2) add w4 to main
I think that's all the changes

i did this in a hack-ey way so I don't want to push my changes


'''
def run_ca_on_GA():
	agents = [GA() for i in range(20)]
	

	gen = 0
	while gen < 25:

		print('generation: ',gen)
		for a in agents:
			a.fitness = ca.main(grid=a.board, time_steps=10,gif_on=False,individual_fitness=False)

		agents = sorted(agents,key=lambda x: x.fitness,reverse=True)
		#have not tinkered with the top-n cutoff for each generation
		#or with creating new random agents to keep diversity up
		agents = agents[:4]
		children = generate_generation(agents)
		print(agents[0].fitness) #current max fitness
		agents.extend(children)
		gen += 1



if __name__ == '__main__':
	#run_ca_on_GA()
	pass