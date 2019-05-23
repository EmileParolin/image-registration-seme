# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:40:39 2019

@author: mehdi
"""

from get_image import getImage
from transport import get_pi, get_t
n_s = 250
Ximg, Ximg_black, Ximg_s = getImage(r"C:\Users\mehdi\Desktop\test\rectangle0.png",
                                    n_s = n_s)
Yimg, Yimg_black, Yimg_s = getImage(r"C:\Users\mehdi\Desktop\test\rectangle1.png", 
                                    n_s = n_s)

xij_s, X_s = Ximg_s
yij_s, Y_s = Yimg_s

xij, X = Ximg_black
yij, Y = Yimg_black

res, pi = get_pi(xij_s, yij_s)
A, b = get_t(xij_s, yij_s, pi)


zij = np.dot(A[na,:,:], xij.T).T[:,:,0] + b[na,:]


plt.scatter(xij[:,0], xij[:, 1])
plt.scatter(yij[:,0], yij[:, 1])
plt.scatter(zij[:,0], zij[:, 1])