import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
import os
import argparse
import pickle
from copy import deepcopy
from scipy.spatial.distance import pdist
from collections import defaultdict

def fitness(data, weight_vector=[0.33,0.33,0.33]):
    result = deepcopy(data[0])

    f1, f2 , f3 = -1, -1, -1

    add1 = np.vectorize(lambda x: x+1 if x > 0 else 0)
    for d in range(1,len(data)):
        result = add1(result)

        for i in range(data[d].shape[0]):
            for j in range(data[d].shape[0]):
                if data[d][i,j] > 0:
                    result[i,j] = 1

        f1 = max(f1, np.max(result))

        indices = np.transpose(np.nonzero(result))


        bins = defaultdict(list)
        for index in indices:
            bins[result[index[0], index[1]]].append(index)

        max_oscillating_cells = (-1,-1) #   (key, num oscillating cells)
        for key, vals in bins.items():
            if len(vals) > max_oscillating_cells[1]:
                max_oscillating_cells = (key, len(vals))


        if max_oscillating_cells[1] > f2:
            f2 = max_oscillating_cells[1]

            dist = pdist(bins[max_oscillating_cells[0]])
            f3 = 1 / (dist.sum()/max_oscillating_cells[1])


    #normalize the fitness values to range [0,1]
    f1 = f1 / (len(data) - 1)
    f2 = f2 / (result.shape[0]*result.shape[1])
    #f3 is implicity normalized

    return np.dot(weight_vector, [f1,f2,f3])

def num_neighbors(grid, i, j):
    return grid[(i - 1) % grid.shape[0], j] + grid[(i - 1) % grid.shape[0], (j - 1) % grid.shape[1]] + grid[
        (i - 1) % grid.shape[0], (j + 1) % grid.shape[1]] + grid[(i + 1) % grid.shape[0], j] + grid[
               (i + 1) % grid.shape[0], (j - 1) % grid.shape[1]] + grid[
               (i + 1) % grid.shape[0], (j + 1) % grid.shape[1]] + grid[i, (j - 1) % grid.shape[1]] + grid[
               i, (j + 1) % grid.shape[1]]


def add_neighbors(temp_set, i, j, x_shape, y_shape):
    temp_set.add(((i - 1) % x_shape, (j - 1) % y_shape))
    temp_set.add(((i - 1) % x_shape, j))
    temp_set.add(((i - 1) % x_shape, (j + 1) % y_shape))
    temp_set.add((i, j))
    temp_set.add((i, j))
    temp_set.add(((i + 1) % x_shape, (j - 1) % y_shape))
    temp_set.add(((i + 1) % x_shape, j))
    temp_set.add(((i + 1) % x_shape, (j + 1) % y_shape))
    return temp_set

def plot_grid(game_grid, filename):
    #plt.figure(figsize=(10,10))
    plt.imshow(game_grid, cmap='binary')
    plt.gca().set_xticks(np.arange(-.5, game_grid.shape[0], 1))
    plt.gca().set_yticks(np.arange(-.5, game_grid.shape[1], 1))
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.grid(linewidth=2)

    plt.savefig(filename)
    plt.clf()
    #plt.close()
    return imageio.imread(filename+'.png')

def game_of_life(game_grid, time_steps, filename, fitness_weights, delete_pngs, gif_on):

    images = []
    cells_to_update = []
    fitness_frames = []
    for step in range(time_steps):
        fitness_frames.append(game_grid)
        if gif_on:
            result = plot_grid(game_grid, filename+str(step))
            images.append(result)

        update_grid = np.zeros(game_grid.shape)

        if step == 0:
            cells_to_update = [(i,j) for i in range(game_grid.shape[0]) for j in range(game_grid.shape[1])]   #all cells in the grid are added to be checked because we do not know anything about the cells at the start

        temp = set()
        for tup in cells_to_update:
            i, j = tup[0], tup[1]
            nn = num_neighbors(game_grid, i, j)

            if nn == 3 or (game_grid[i,j] == 1 and nn == 2):
                update_grid[i, j] = 1
                temp = add_neighbors(temp, i, j, game_grid.shape[0], game_grid.shape[1])

        game_grid = update_grid
        cells_to_update = list(temp)

    if gif_on:
        imageio.mimsave(filename+'.gif', images)

    if delete_pngs:
        for step in range(time_steps):
            os.remove(filename+str(step)+'.png')


    # with open('frames.p','wb') as pFile:
        # pickle.dump(fitness_frames,pFile)

    return fitness(fitness_frames, fitness_weights)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Game of Life')
    parser.add_argument('--n', type=int, default=50,
                        help='size of grid (default: 50)')
    parser.add_argument('--timesteps', type=int, default=19,
                        help='number of time steps (default: 19)')
    parser.add_argument('--filename', type=str, default='gliders',
                        help='filename for oscillator type (default: gliders)')
    parser.add_argument('--deletepngs', type=bool, default=True,
                        help='delete the pngs for each time step (default: True)')
    parser.add_argument('--w1', type=int, default=0.33,
                        help='weight for fitness function 1')
    parser.add_argument('--w2', type=int, default=0.33,
                        help='weight for fitness function 2')
    parser.add_argument('--w3', type=int, default=0.33,
                        help='weight for fitness function 3')
    parser.add_argument('--gif', type=int, default=1,
                        help='should we create a gif')

    args = parser.parse_args()

    n = args.n
    time_steps = args.timesteps
    filename = 'oscillators/' + args.filename
    delete_pngs = args.deletepngs
    gif_on = args.gif

    if not gif_on:
        #no pngs to delete
        delete_pngs = False


    game_grid = np.zeros((n,n))

    weights = [args.w1, args.w2, args.w3]

    #define initial seed

    # with open('./seeds/glider.p','rb') as pFile:
    #     game_grid = pickle.load(pFile)
    # for i in range(game_grid.shape[0]):
    #     for j in range(game_grid.shape[0]):
    #         if np.random.uniform(0,1) > 0.5:
    #             game_grid[i,j] = 1
    
    with open('./seeds/penta-decathlon.p','rb') as pFile:
        game_grid = pickle.load(pFile)

    # with open('./seeds/penta-decathlon.p','wb') as pFile:
    #     pickle.dump(game_grid,pFile)
    if not os.path.isdir('oscillators'):
        os.makedirs('oscillators')


    fitness = game_of_life(game_grid, time_steps, filename, weights, delete_pngs, gif_on)

    print('Fitness: ',fitness)




