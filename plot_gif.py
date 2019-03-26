#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:05:21 2019

@author: joshladd
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio
import pickle


#simulates game for t_steps on given board
#fs = figsize
#path is location of existing folder
#name is for name of files
#will output a png for each timestep so output to a subdirectory
def plot_gif(t_steps,board,fs=5,path='./',name=''):
    images = []
    plt.imshow(board)

    f = plt.figure(figsize=(fs,fs))
    for t in range(t_steps):
        plt.imshow(board,cmap='binary')
        filename = path+str(t)+name+'.png'
        plt.savefig(filename)
        plt.clf()
        images.append(imageio.imread(filename))
        board = nextBoard(board)  #whatever update function we use for the board
    plt.figure()
    plt.imshow(board)
    imageio.mimsave(path + name + str(t_steps) + '.gif', images)
    
    
    
    
    
def main():
    
    with open('./oscillators/penta-decathlon.p','rb') as pFile:
        board = pickle.load(pFile)
        
    plot_gif(15,board,path='./penta-decathlon/')
    
    
    
if __name__ == '__main__':
    main()