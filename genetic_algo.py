import cellular_automaton as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from os import cpu_count
import os
import test_fitness



####################################
#Global variables
weights = []

class GA:
    def __init__(self, board_size=(50,50), ratio=0.2, fitness=0, board=[], gauss_init=False):
        self.fitness = fitness
        if board == []:
            if gauss_init:
                game_grid = np.zeros(board_size)
                xs,ys = np.random.multivariate_normal([board_size[0]/2, board_size[1]/2], [[board_size[0]/5, 0], [0,board_size[1]/5]], int(board_size[0]*board_size[1]*ratio)).T
                xs = [int(x) for x in xs]
                ys = [int(y) for y in ys]
                for x,y in zip(xs, ys):
                    game_grid[x,y] = 1
                self.board = game_grid
            else:
                self.board = np.random.binomial(n=1, p=ratio, size=board_size)
        else:
            self.board = board

    def crossover(self, partner):
        #TODO consider multiple partners
        chromosome1 = np.ravel(self.board)
        chromosome2 = np.ravel(partner.board)
        splt = np.random.randint(len(chromosome1))
        child = np.concatenate((chromosome1[:splt], chromosome2[splt:]))

        if np.array_equal(self.board, partner.board) or np.random.uniform(0,1) < 0.1:
            idx = np.random.randint(len(child))
            child[idx] = 0 if child[idx] else 1 #flip the bit

        child = np.reshape(child, self.board.shape)
        return GA(board=child)


def run_ca(agent):
    global weights
    #weights was defined as a global variable to the script to get fix the pool.map function
    f = ca.run_for_ga(game_grid=agent.board, fitness_weights=weights, time_steps=22)
    agent.fitness = f
    return agent


def run_genetic_algorithm(max_num_generations=1000, fitness_threshold=1, population_size=124, top_k=5, num_to_return=5, save_every_n=10):
    agents = [GA() for _ in range(population_size)]

    global weights
    weights = test_fitness.get_weights()

    for g in range(max_num_generations):
        with Pool(processes=cpu_count()) as pool:
            agents = pool.map(run_ca, agents)

        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
        agents = agents[:-int(top_k*(top_k-1))] #each of the top_k agents will breed with eachother, thus we will remove the bottom n*n-1 agents

        if agents[0].fitness >= fitness_threshold:  #TODO or the GA has stopped improving
            break   #we are done

        children = []
        for i in range(top_k):
            for j in range(top_k):
                if i != j:
                    children.append(agents[i].crossover(agents[j]))

        agents.extend(children)

        print('Generation %s, best fitness %s'%(g, agents[0].fitness))

        if g%save_every_n == 0:
            for num, r in enumerate(agents[:num_to_return]):
                filename = 'ga_results/ga'+str(num)
                with open(filename, 'wb') as pFile:
                    pickle.dump(r.board, pFile)

    return agents[:num_to_return]

def ga_best_performers(max_num_generations=1000, fitness_threshold=1, top_k=10, num_to_return=5, ratio=0.1, save_every_n=10):
    population_size = top_k**2 + top_k
    agents = [GA(ratio=ratio) for _ in range(population_size)]

    global weights
    weights = test_fitness.get_weights()

    for g in range(max_num_generations):
        with Pool(processes=cpu_count()) as pool:
            agents = pool.map(run_ca, agents)

        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
        agents = agents[:top_k] #keep the top_k performers

        if agents[0].fitness >= fitness_threshold:  #TODO or the GA has stopped improving
            break   #we are done

        children = []
        for i in range(top_k):
            for j in range(top_k):
                    children.append(agents[i].crossover(agents[j]))

        agents.extend(children)

        print('Generation %s, best fitness %s'%(g, agents[0].fitness))

        if g%save_every_n == 0:
            for num, r in enumerate(agents[:num_to_return]):
                filename = 'ga_results/bp' + str(num)
                with open(filename, 'wb') as pFile:
                    pickle.dump(r.board, pFile)


    return agents[:num_to_return]

def ga_best_performers_with_noise(max_num_generations=1000, fitness_threshold=1, top_k=10, num_to_return=5, save_every_n=10):
    population_size = top_k**2 + top_k + top_k  #we will keep the top_k and a random number of k agents
    agents = [GA() for _ in range(population_size)]

    global weights
    weights = test_fitness.get_weights()

    for g in range(max_num_generations):
        with Pool(processes=cpu_count()) as pool:
            agents = pool.map(run_ca, agents)

        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)

        nums = np.random.randint(low=top_k, high=len(agents), size=top_k)

        agents = agents[:top_k] + [agents[i] for i in nums]

        if agents[0].fitness >= fitness_threshold:  #TODO or the GA has stopped improving
            break   #we are done

        children = []
        for i in range(top_k):
            for j in range(top_k):
                    children.append(agents[i].crossover(agents[j]))

        agents.extend(children)

        print('Generation %s, best fitness %s'%(g, agents[0].fitness))

        if g%save_every_n == 0:
            for num, r in enumerate(agents[:num_to_return]):
                filename = 'ga_results/bpn' + str(num)
                with open(filename, 'wb') as pFile:
                    pickle.dump(r.board, pFile)

    return agents[:num_to_return]

def plot_grid(game_grid):

    plt.imshow(game_grid, cmap='binary')
    plt.gca().set_xticks(np.arange(-.5, game_grid.shape[0], 1))
    plt.gca().set_yticks(np.arange(-.5, game_grid.shape[1], 1))
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.grid(linewidth=2)
    plt.show()


if __name__ == '__main__':
    if not os.path.isdir('ga_results'):
        os.makedirs('ga_results')

    # result = run_genetic_algorithm()
    # result = ga_best_performers()
    result = ga_best_performers_with_noise()

    for num, r in enumerate(result):
        filename = 'ga_results/'+str(num)
        with open(filename, 'wb') as pFile:
            pickle.dump(r.board, pFile)





