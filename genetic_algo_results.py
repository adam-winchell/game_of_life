import cellular_automaton as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from os import cpu_count
import os
import test_fitness
import argparse
from functools import partial


####################################
# Global variables
#used to avoid passing multiple args to multiprocess
time_steps = 11
fitness_funcs = 'f1f2f6f7'
####################################

class GA:
    def __init__(self, board_size=(50,50), ratio=0.2, fitness=0, board=[], gauss_init=False, num_bits_to_flip=[1,2,3]):
        self.fitness = fitness
        self.num_bits_to_flip = num_bits_to_flip
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
        child1 = np.concatenate((chromosome1[:splt], chromosome2[splt:]))
        child2 = np.concatenate((chromosome2[:splt], chromosome1[splt:]))

        if np.array_equal(self.board, partner.board) or np.random.uniform(0,1) < 0.1:
            num = np.random.choice(self.num_bits_to_flip)
            for _ in range(num):
                idx = np.random.randint(len(child1))
                child1[idx] = 0 if child1[idx] else 1 #flip the bit

        if np.array_equal(self.board, partner.board) or np.random.uniform(0,1) < 0.1:
            num = np.random.choice(self.num_bits_to_flip)
            for _ in range(num):
                idx = np.random.randint(len(child2))
                child2[idx] = 0 if child2[idx] else 1 #flip the bit

        child1 = np.reshape(child1, self.board.shape)
        child2 = np.reshape(child2, self.board.shape)
        return [GA(board=child1), GA(board=child2)]


def run_ca(agent,funcs=['f1','f2','f6','f7'],weights=[]):
    if len(weights) == 0:
        # print('here')
        weights = [1/len(funcs) for i in funcs]


    # print(funcs)
    #weights was defined as a global variable to the script to get fix the pool.map function
    f = ca.run_for_ga(game_grid=agent.board, fitness_weights=weights, time_steps=time_steps,funcs=funcs)
    agent.fitness = f
    return agent


def save_agents(agents, filename):
    for num, r in enumerate(agents):
        f = 'ga_results/'+filename +'_'+ fitness_funcs+'_'+str(num)
        with open(f, 'wb') as pFile:
            pickle.dump(r.board, pFile)

def run_genetic_algorithm(max_num_generations=1000, fitness_threshold=2, population_size=124, top_k=5, num_to_return=5, save_every_n=10):
    agents = [GA() for _ in range(population_size)]

    # global weights
    # weights = test_fitness.get_weights()

    for g in range(max_num_generations):
        with Pool(processes=cpu_count()) as pool:
            agents = pool.map(run_ca, agents)

        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)

        if g % save_every_n == 0:
            save_agents(agents[:num_to_return], 'ga')

        agents = agents[:-int(top_k*(top_k-1))] #each of the top_k agents will breed with eachother, thus we will remove the bottom n*n-1 agents

        if agents[0].fitness >= fitness_threshold:  #TODO or the GA has stopped improving
            break   #we are done

        children = []
        for i in range(top_k):
            for j in range(top_k):
                if i != j:
                    children.extend(agents[i].crossover(agents[j]))

        agents.extend(children)

        print('Generation %s, best fitness %s'%(g, agents[0].fitness))



    return agents[:num_to_return]

def ga_best_performers(weights,max_num_generations=1000, fitness_threshold=2, top_k=10, num_to_return=5, ratio=0.1, save_every_n=10,funcs=['f1','f2','f6','f7']):
    population_size = top_k**2 + top_k
    agents = [GA(ratio=ratio) for _ in range(population_size)]
    run_results = partial(run_ca,funcs=funcs,weights=weights)

    for g in range(max_num_generations):
        with Pool(processes=cpu_count()) as pool:
            agents = pool.map(run_results, agents)

        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)

        if g % save_every_n == 0:
            save_agents(agents[:num_to_return], 'bp')

        agents = agents[:top_k] #keep the top_k performers

        if agents[0].fitness >= fitness_threshold:  #TODO or the GA has stopped improving
            break   #we are done

        children = []
        for i in range(top_k):
            for j in range(top_k):
                    children.extend(agents[i].crossover(agents[j]))

        agents.extend(children)

        print('Generation %s, best fitness %s'%(g, agents[0].fitness))




    return agents[:num_to_return]

def ga_best_performers_with_noise(weights,max_num_generations=1000, fitness_threshold=2, top_k=10, num_to_return=5, save_every_n=10, ratio=0.1,funcs=['f1','f2','f6','f7']):
    population_size = (2*top_k)**2 + top_k + top_k  #we will keep the top_k and a random number of k agents and each round we reproduce the the top_k + a random_k, everyone with eachother
    agents = [GA(ratio=ratio) for _ in range(population_size)]
    run_results = partial(run_ca,funcs=funcs,weights=weights)
    for g in range(max_num_generations):
        with Pool(processes=cpu_count()) as pool:
            agents = pool.map(run_results, agents)

        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)

        if g % save_every_n == 0:
            save_agents(agents[:num_to_return], 'bpn')

        if agents[0].fitness >= fitness_threshold:  #TODO or the GA has stopped improving
            break   #we are done

        nums = np.random.randint(low=top_k, high=len(agents), size=top_k)

        agents = agents[:top_k] + [agents[i] for i in nums]


        children = []
        for i in range(len(agents)):
            for j in range(len(agents)):
                    children.extend(agents[i].crossover(agents[j]))

        agents.extend(children)

        print('Generation %s, best fitness %s'%(g, agents[0].fitness))

    return agents[:num_to_return]



def ga_weighted_best_performers(weights,max_num_generations=1000, fitness_threshold=2, num_parents=10, num_to_return=5, save_every_n=10, ratio=0.1,funcs=['f1','f2','f6','f7']):
    population_size = num_parents**2 + num_parents
    agents = [GA(ratio=0.1) for _ in range(population_size)]
    run_results = partial(run_ca,funcs=funcs,weights=weights)
    # print(funcs)
    for g in range(max_num_generations):
        with Pool(processes=cpu_count()) as pool:
            agents = pool.map(run_results, agents)

        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)

        if g % save_every_n == 0:
            save_agents(agents[:num_to_return], 'wbp')

        if agents[0].fitness >= fitness_threshold:  #TODO or the GA has stopped improving
            break   #we are done

        probabilities = [((len(agents) - agent_rank)**2 )/ ((len(agents) * (len(agents) + 1)*(2*len(agents) + 1)) / 6) for agent_rank in range(len(agents))]    #based on sum of i to k of k^2

        agents = np.random.choice(a=agents, p=probabilities, replace=False, size=num_parents)

        children = []
        for i in range(len(agents)):
            for j in range(len(agents)):
                    children.extend(agents[i].crossover(agents[j]))

        agents = [agent for agent in agents]
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
    if not os.path.isdir('ga_results'):
        os.makedirs('ga_results')

    parser = argparse.ArgumentParser(description='Genetic Algorithm')
    parser.add_argument('--algo', type=str, default='ga',
                        help='which ga algorithm to run  (default: bp)')

    args = parser.parse_args()

    # if args.algo == 'ga':
    #     result = run_genetic_algorithm()
    # elif args.algo == 'wbp':
    #     global weights 
    #     weights = test_fitness.get_weights(funcs=['f1','f2','f6'])
    #     result = ga_best_performers(funcs =['f1','f2','f6'])
    #     # result = ga_weighted_best_performers(max_num_generations=10000)
    # elif args.algo == 'bp':
    #     result = ga_best_performers(max_num_generations=10000, top_k=15)
    # elif args.algo == 'bpn':
    #     result = ga_best_performers_with_noise(max_num_generations=10000)
    # elif args.algo == 'results':
    #     funcs = ['f1','f2','f6']
    #     fitness_funcs = ''.join(funcs)
    #     weights = test_fitness.get_weights(funcs=funcs)
    #     result = ga_best_performers(max_num_generations=11,funcs=funcs)


    solo = [['f1'],['f2'],['f6'],['f7']]
    leave_one_out = [['f1','f2','f7'],['f1','f2','f6'],['f1','f6','f7'],['f2','f6','f7']]
    # populations = [ga_weighted_best_performers,ga_best_performers,ga_best_performers_with_noise]
    populations = [ga_best_performers_with_noise]

    funcs_test = solo + leave_one_out
    for item in funcs_test:
        weights = test_fitness.get_weights(funcs=item)
        fitness_funcs = ''.join(item)
        print(item,weights)
        for p in populations:
            print(p.__name__)
            result = p(weights,max_num_generations=1000,funcs=item)



    #TODO get graph of fitness over time

    save_agents(result, args.algo)






