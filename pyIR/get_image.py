import numpy as np
from PIL import Image


def getImage(path, threshold = .2, n_s = 250):
    # Obtaining data
    x=Image.open(path,'r')
    xBW=x.convert('L')
    xRGB=x.convert('RGB')
    yBW=np.asarray(xBW.getdata(),dtype=np.float64)
    yRGB=np.asarray(xRGB.getdata(),dtype=np.float64)
    width, height = xBW.size
    # Line and column indices
    i = np.repeat(np.linspace(1, height, height), width)
    j = np.tile(np.linspace(1, width, width), reps=height)
    # Matrix of complete coordinates
    ij = np.vstack((i, j)).T
    # Points above the threshold
    black = yBW[yBW < threshold * np.mean(yBW)]
    i_black = i[yBW < threshold * np.mean(yBW)]
    j_black = j[yBW < threshold * np.mean(yBW)]
    # Matrix of black coordinates
    ij_black = np.vstack((i_black, j_black)).T
    # Sampling without replacement
    sampled = np.random.choice(np.arange(1, black.shape[0], 1),
                               size=min(n_s, black.shape[0]-1), replace=False)
    # Matrix of sampled black coordinates
    ij_s = ij_black[sampled]
    black_s = black[sampled]
    # Returns both the matrix and the intensity with and without the sampling
    return ((ij, yBW, yRGB), (ij_black, black), (ij_s, black_s))

