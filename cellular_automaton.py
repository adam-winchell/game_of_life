import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def game_of_life(grid, time_steps):

    def num_neighbors(grid, i, j):
        return grid[(i - 1) % grid.shape[0], j] + grid[(i - 1) % grid.shape[0], (j - 1) % grid.shape[1]] + grid[
            (i - 1) % grid.shape[0], (j + 1) % grid.shape[1]] + grid[(i + 1) % grid.shape[0], j] + grid[
                   (i + 1) % grid.shape[0], (j - 1) % grid.shape[1]] + grid[
                   (i + 1) % grid.shape[0], (j + 1) % grid.shape[1]] + grid[i, (j - 1) % grid.shape[1]] + grid[
                   i, (j + 1) % grid.shape[1]]

    cmap = ListedColormap(['k', 'w'])

    plt.ion()  # makes the plot interactive
    for step in range(time_steps):
        plt.matshow(grid, cmap=cmap)
        plt.pause(0.2)
        update_grid = np.zeros(grid.shape)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                nn = num_neighbors(grid, i, j)

                if grid[i, j] == 1:
                    if nn == 2 or nn == 3:
                        update_grid[i, j] = 1
                    else:  # under and over population
                        update_grid[i, j] = 0
                elif nn == 3:  # reproduction
                    update_grid[i, j] = 1

        grid = update_grid


if __name__ == "__main__":

    grid = np.zeros((10,10))

    grid[3,3] = 1
    grid[3, 4] = 1
    grid[3, 5] = 1
    grid[2, 5] = 1
    grid[1, 4] = 1

    game_of_life(grid, 10)




