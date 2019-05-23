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

source = "../images/Cas1_lung_biopsy/HLA.png"
target = "../images/Cas1_lung_biopsy/cd.png"

source = "../images/test/cat0.png"
target = "../images/test/cat4.png"

source = "../images/test/malevich0.png"
target = "../images/test/malevich2.png"

source = "../images/test/pacman0.png"
target = "../images/test/pacman3.png"

source = "../images/test/rectangle0.png"
target = "../images/test/rectangle1.png"

# Parameters
n_s = 50 # Number of sampling points
threshold = .2 # Pixel detection threshold
Ng = 3 # Number of gaussians
# Image models
Ximg, Ximg_black, Ximg_s = getImage(source, n_s=n_s, threshold=threshold)
Yimg, Yimg_black, Yimg_s = getImage(target, n_s=n_s, threshold=threshold)
# Position and color of all image pixels
xij, X = Ximg
yij, Y = Yimg
# Position and color of image pixels above threshold
xij_black, X_black = Ximg_black
yij_black, Y_black = Yimg_black
# Position and color of image pixels above threshold sample subset
xij_s, X_s = Ximg_s
yij_s, Y_s = Yimg_s
# Translation of target image
Xmax = np.max(xij[:,0])
yij[:,0] += Xmax
yij_black[:,0] += Xmax
yij_s[:,0] += Xmax

if False :
    # Position and color of sample image pixels
    x = xij_s
    y = yij_s
    # routines for points
    distance = euclidian_distance
    affine_transfo = euclidian_affine_transfo
    pi_prod = points_pi_prod
else :
    # Gaussian fits
    x = fit_gaussians(xij_black, Ng)
    y = fit_gaussians(yij_black, Ng)
    # routines for gaussians
    distance = wasserstein_distance
    affine_transfo = gaussian_affine_transfo
    pi_prod = gaussian_pi_prod

# Solve optimal transport problem
res, pi = get_pi(x, y, distance)
# Compute affine transformation
sol, A, b = get_t(x, y, pi, distance, affine_transfo, pi_prod)
# Apply affine transformation to image pixels
zij_black = np.dot(A[na,:,:], xij_black.T).T[:,:,0] + b[na,:]
# Plot
plt.cla()
plt.scatter(xij_black[:,0], xij_black[:,1], marker=".")
plt.scatter(yij_black[:,0], yij_black[:,1], marker="x")
plt.scatter(zij_black[:,0], zij_black[:,1], marker="+")
