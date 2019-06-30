import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_image(path):
    """
    Load image and computes pixels indices.
    """
    # Obtaining data
    x = Image.open(path, 'r')
    # Array size
    width, height = x.size
    # Black and White
    xBW = np.asarray(x.convert('L').getdata(), dtype=np.float64)
    # RGB colors
    xRGB = np.asarray(x.convert('RGB').getdata(), dtype=np.float64)
    # Line and column indices
    i = np.tile(np.linspace(1, width, width), reps=height)
    j = height - np.repeat(np.linspace(1, height, height), width)
    # Matrix of complete coordinates
    ij = np.vstack((i, j)).T
    return ij, xBW, xRGB


def points_above_threshold(ij, xBW, threshold = .2):
    """
    Select only pixel coordinates with intensity above threshold value.
    """
    # Indices of points with intensity above threshold
    selection = xBW < threshold * np.mean(xBW)
    # Matrix of black coordinates
    return ij[selection,:]
