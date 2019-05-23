# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:53:38 2019

"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def getImage(path, threshold = .2, n_s = 250):    
    x=Image.open(path,'r')
    x=x.convert('L') 
    y=np.asarray(x.getdata(),dtype=np.float64)
    width, height = x.size
    
    # Line and column indices
    i = np.repeat(np.linspace(1, height, height), width)
    j = np.tile(np.linspace(1, width, width), reps=height)
    
    # Matrix of complete coordinates
    ij = np.vstack((i, j)).T
    
    # Points above the threshold
    black = y[y < threshold*np.mean(y)]
    i_black = i[y < threshold*np.mean(y)]
    j_black = j[y < threshold*np.mean(y)]
    
    # Matrix of black coordinates
    ij_black = np.vstack((i_black, j_black)).T
    
    # Sampling without replacement
    sampled = np.random.choice(np.arange(1, black.shape[0], 1),
                               size=min(n_s, black.shape[0]-1), replace=False)
    
    # Matrix of sampled black coordinates
    ij_s = ij_black[sampled]
    black_s = black[sampled]
    
    # Returns both the matrix and the intensity with and without the sampling
    return ((ij, y), (ij_black, black), (ij_s, black_s))

