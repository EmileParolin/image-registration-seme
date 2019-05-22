# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:03:51 2019

@author: mehdi
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Lecture d'une image
x=Image.open("cd.png",'r')
x=x.convert('L') 
y=np.asarray(x.getdata(),dtype=np.float64)
width, height = x.size

# Line and column indices
i = np.repeat(np.linspace(1, height, height), width)
j = np.tile(np.linspace(1, width, width), reps=height)

# Matrix of coordinates
X = np.vstack((i, j)).T

# Transformation matrix
A = np.matrix([[1, 1], [-1, 1]])
b = np.array([0, 0])
# Transformation 
Tx = np.array(np.dot(A, X.T).T + b.reshape(1,2))

# Affichage
plt.scatter(Tx[:,0], Tx[:, 1], c = y)