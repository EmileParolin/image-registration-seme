import numpy as np
import scipy
from scipy.linalg import sqrtm


def wasserstein_distance(gauss1, gauss2):
    """
    Wasserstein distance between two gaussians.
    """
    mu1 = gauss1[0]
    mu2 = gauss2[0]

    cov1 = gauss1[1]
    cov2 = gauss2[1]

    sigma = cov1 + cov2 - 2 * sqrtm(sqrtm(cov1) @ cov2 @ sqrtm(cov1))
    dist = np.linalg.norm(mu1 - mu2)**2 + np.matrix.trace(sigma)
    return dist


def gaussian_transfo(A, b, y):
    prod = []
    for i in range(len(y)):
        prod.append((A @ y[i][0] + b, A @ y[i][1] @ A.T))
    return prod


def gaussian_pi_prod(pi, y):   
    prod_mean = []
    prod_var = []

    for i in range(len(y)):
        prod_m = 0
        prod_v = 0
        for j in range(len(y)):
            prod_m += (pi[i,j]) * y[j][0]
            prod_v += ((pi[i,j])**2) * y[j][1]

        prod_mean.append(prod_m)
        prod_var.append(prod_v)

    
    return [(prod_mean[i], prod_var[i]) for i in range(len(y))]
