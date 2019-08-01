import numpy as np
import matplotlib.pyplot as plt

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pyIR import *

def test_code(postfix):
    # Fake source points
    N = 25
    xa = np.vstack((np.random.normal(18,1,N), np.random.normal(18,10,N))).T
    xb = np.vstack((np.random.normal(29,9,N), np.random.normal(3,0.5,N))).T
    x = np.vstack((xa, xb))
    # Fake target points
    y = np.array([euclidian_affine_transfo(A0, b0, x[i,:]) for i in range(x.shape[0])])
    # Transfo
    A, b, res, sol, pi = get_transfo(x, y, euclidian_distance,
            euclidian_affine_transfo, points_pi_prod)
    z = apply_t(A, b, x)
    # Error
    err = np.linalg.norm(z - y)
    print(err)
    # Plot
    plot_images(A, b, x, y)
    plt.savefig("./outputs/test_points_"+postfix+".png")

    # Fake source Gaussian mixture
    N = 4
    ma = np.vstack((np.random.normal(18,1,N), np.random.normal(18,10,N))).T
    mb = np.vstack((np.random.normal(29,9,N), np.random.normal(3,0.5,N))).T
    mx = np.vstack((ma, mb))
    Sx = [5*(np.random.rand(2,2)-0.5) for i in range(2*N)]
    Sx = [np.dot(s, s.T) for s in Sx] # symmetric positive semi-definite covariance matrices
    x = [(mx[i,:],Sx[i]) for i in range(2*N)]
    # Fake target Gaussian mixture
    y = [gaussian_affine_transfo(A0, b0, x[i]) for i in range(len(x))]
    my = apply_t(A0, b0, mx)
    # Transfo
    A, b, res, sol, pi = get_transfo(x, y, wasserstein_distance,
            gaussian_affine_transfo, gaussian_pi_prod)
    z = [gaussian_affine_transfo(A, b, g) for g in x]
    mz = apply_t(A, b, mx)
    # Error
    err = np.linalg.norm(mz - my)
    print(err)
    # Plot
    plot_images(A, b, mx, my)
    plot_ellipses(x, c='black')
    plot_ellipses(y, c='red', thickness=2)
    plot_ellipses(z, c='blue')
    plt.savefig("./outputs/test_gaussians_"+postfix+".png")


# Random affine transformation: does not work well
A0 = 4 * (np.random.rand(2,2)-0.5)
b0 = 12 * (np.random.rand(2)-0.5)
test_code("random")

# Affine transformation with "small" rotation: works almost every time
A0 = 4 * (np.random.rand(2,2)-0.5) + 10 * np.diag(np.random.rand(2))
b0 = 12 * (np.random.rand(2)-0.5)
test_code("small_rot")
