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
