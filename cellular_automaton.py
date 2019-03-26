import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
import os
import argparse


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
    plt.imshow(game_grid, cmap='binary')
    plt.gca().set_xticks(np.arange(-.5, game_grid.shape[0], 1))
    plt.gca().set_yticks(np.arange(-.5, game_grid.shape[1], 1))
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.grid(linewidth=2)

    plt.savefig(filename)
    plt.clf()

    return imageio.imread(filename+'.png')

def game_of_life(game_grid, time_steps, filename, delete_pngs=True):

    images = []
    cells_to_update = []
    for step in range(time_steps):
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


    imageio.mimsave(filename+'.gif', images)

    if delete_pngs:
        for step in range(time_steps):
            os.remove(filename+str(step)+'.png')




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

    args = parser.parse_args()

    n = args.n
    time_steps = args.timesteps
    filename = 'oscillators/' + args.filename
    delete_pngs = args.deletepngs

    game_grid = np.zeros((n,n))

    #define initial seed
    game_grid[1,1] = 1
    game_grid[1,3] = 1
    game_grid[2,3] = 1
    game_grid[2,2] = 1
    game_grid[3,2] = 1

    if not os.path.isdir('oscillators'):
        os.makedirs('oscillators')


    game_of_life(game_grid, time_steps, 'oscillators/glider', delete_pngs)




