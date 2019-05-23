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


# routines for points
distance = euclidian_distance
affine_transfo = euclidian_affine_transfo
pi_prod = points_pi_prod

def get_transfo(x, y):
    # Solve optimal transport problem
    res, pi = get_pi(x, y, distance)
    # Compute affine transformation
    A, b = get_t(x, y, pi, distance, affine_transfo, pi_prod)
    # Apply affine transformation to image pixels
    z = np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:]
    # Plots
    plt.cla()
    plt.scatter(x[:,0], x[:, 1], marker=".")
    plt.scatter(y[:,0], y[:, 1], marker="x")
    plt.scatter(z[:,0], z[:, 1], marker="+")
    return res, pi, A, b, z

if False :

    # Generating some random points
    N = 50
    x = np.vstack((np.random.normal(0,1,N), np.random.normal(0,10,N))).T
    y = np.vstack((np.random.normal(10,10,N), np.random.normal(3,2,N))).T
    res, pi, A, b, z = get_transfo(x, y)

    # Generating some random points (cross)
    N = 25
    x1 = np.vstack((np.random.normal(0,1,N), np.random.normal(0,10,N))).T
    x2 = np.vstack((np.random.normal(12,7,N), np.random.normal(-15,0.5,N))).T
    y1 = np.vstack((np.random.normal(10,10,N), np.random.normal(-4,2,N))).T
    y2 = np.vstack((np.random.normal(0,0.1,N), np.random.normal(3,8,N))).T
    x = np.vstack((x1, x2))
    y = np.vstack((y1, y2))
    res, pi, A, b, z = get_transfo(x, y)

    # y = x
    N = 50
    x = np.vstack((np.random.normal(0,1,N), np.random.normal(0,10,N))).T
    res, pi, A, b, z = get_transfo(x, x)

    # Prescribed affine transformation A = Id, b random
    N = 50
    A0 = np.eye(2)
    b0 = 8 * np.random.rand(2)
    x = np.vstack((np.random.normal(0,1,N), np.random.normal(0,10,N))).T
    y = affine_transfo(A0, b0, x)
    res, pi, A, b, z = get_transfo(x, y)

    # Prescribed affine transformation A = a Id, b zero
    N = 50
    A0 = np.eye(2)
    A0[1,1] = 3
    A0[2,2] = 7
    b0 = np.zeros(2, dtype=np.float64)
    x = np.vstack((np.random.normal(0,1,N), np.random.normal(0,10,N))).T
    y = affine_transfo(A0, b0, x)
    res, pi, A, b, z = get_transfo(x, y)

    # Prescribed affine transformation: A, b random
    N = 50
    A0 = 3 * np.random.rand(2,2)
    b0 = 8 * np.random.rand(2)
    x = np.vstack((np.random.normal(0,1,N), np.random.normal(0,10,N))).T
    y = affine_transfo(A0, b0, x)
    res, pi, A, b, z = get_transfo(x, y)
