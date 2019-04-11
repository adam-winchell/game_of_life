import cellular_automaton as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from os import cpu_count

class GA:
    def __init__(self, board_size=(50,50), ratio=0.2, fitness=0, board=[]):
        self.fitness = fitness
        if board == []:
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
    f = ca.run_for_ga(game_grid=agent.board, time_steps=10)
    agent.fitness = f
    return agent


def run_genetic_algorithm(max_num_generations=500, fitness_threshold=0.95, population_size=124, top_k=5, num_to_return=5):
    agents = [GA() for _ in range(population_size)]

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

    return agents[:num_to_return]

def ga_best_performers(max_num_generations=500, fitness_threshold=0.95, top_k=10, num_to_return=5, ratio=0.1, save_every_n=10):
    population_size = top_k**2 + top_k
    agents = [GA(ratio=ratio) for _ in range(population_size)]

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
                filename = 'ga_results/' + str(num)
                with open(filename, 'wb') as pFile:
                    pickle.dump(r.board, pFile)


    return agents[:num_to_return]

def ga_best_performers_with_noise(max_num_generations=500, fitness_threshold=0.95, top_k=10, num_to_return=5):
    population_size = top_k**2 + top_k + top_k  #we will keep the top_k and a random number of k agents
    agents = [GA() for _ in range(population_size)]

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
    # result = run_genetic_algorithm(max_num_generations=1000)
    result = ga_best_performers()
    # result = ga_best_performers_with_noise()

    for num, r in enumerate(result):
        filename = 'ga_results/'+str(num)
        with open(filename, 'wb') as pFile:
            pickle.dump(r.board, pFile)

    plot_grid(result[0].board)



