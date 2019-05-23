import numpy as np
from get_image import getImage

from transport import get_pi
from transport import get_t

from points import euclidian_distance
from points import euclidian_affine_transfo
from points import points_pi_prod

from gaussians import wasserstein_distance

import matplotlib.pyplot as plt
na = np.newaxis

n_s = 50
Ximg, Ximg_black, Ximg_s = getImage("../images/test/rectangle0.png", n_s=n_s)
Yimg, Yimg_black, Yimg_s = getImage("../images/test/rectangle1.png", n_s=n_s)

# Position and color of all image pixels
xij, X = Ximg_black
yij, Y = Yimg_black

if True:
    # Position and color of sample image pixels
    x, X_s = Ximg_s
    y, Y_s = Yimg_s
    # routines for points
    distance = euclidian_distance
    affine_transfo = euclidian_affine_transfo
    pi_prod = points_pi_prod

# Solve optimal transport problem
res, pi = get_pi(x, y, distance)
# Compute affine transformation
A, b = get_t(x, y, pi, distance, affine_transfo, pi_prod)
# Apply affine transformation to image pixels
zij = np.dot(A[na,:,:], xij.T).T[:,:,0] + b[na,:]


plt.scatter(xij[:,0], xij[:, 1])
plt.scatter(yij[:,0], yij[:, 1])
plt.scatter(zij[:,0], zij[:, 1])
