import numpy as np
from get_image import getImage

from transport import get_pi
from transport import get_t

from points import euclidian_distance
from points import euclidian_affine_transfo
from points import points_pi_prod

from gaussians import wasserstein_distance
from gaussians import gaussian_affine_transfo
from gaussians import gaussian_pi_prod
from gaussians import fit_gaussians

import matplotlib.pyplot as plt
na = np.newaxis

n_s = 50
Ximg, Ximg_black, Ximg_s = getImage("../images/test/cat0.png", n_s=n_s)
Yimg, Yimg_black, Yimg_s = getImage("../images/test/cat2.png", n_s=n_s)

# Position and color of all image pixels
xij, X = Ximg_black
yij, Y = Yimg_black

if False :
    # Position and color of sample image pixels
    x, X_s = Ximg_s
    y, Y_s = Yimg_s
    # routines for points
    distance = euclidian_distance
    affine_transfo = euclidian_affine_transfo
    pi_prod = points_pi_prod
else :
    # Position and color of black image pixels
    x_points, X_black = Ximg_black
    y_points, Y_black = Yimg_black
    # Gaussian fits
    Ng = 25
    x = fit_gaussians(x_points, Ng)
    y = fit_gaussians(y_points, Ng)
    # routines for gaussians
    distance = wasserstein_distance
    affine_transfo = gaussian_affine_transfo
    pi_prod = gaussian_pi_prod

# Solve optimal transport problem
res, pi = get_pi(x, y, distance)
# Compute affine transformation
A, b = get_t(x, y, pi, distance, affine_transfo, pi_prod)
# Apply affine transformation to image pixels
zij = np.dot(A[na,:,:], xij.T).T[:,:,0] + b[na,:]


plt.cla()
plt.scatter(xij[:,0], xij[:, 1], marker=".")
plt.scatter(yij[:,0], yij[:, 1], marker="x")
plt.scatter(zij[:,0], zij[:, 1], marker="+")
