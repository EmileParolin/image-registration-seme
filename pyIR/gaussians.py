import numpy as np
import scipy
from scipy.linalg import sqrtm
from sklearn.mixture import GaussianMixture


def wasserstein_distance(gauss1, gauss2):
    """
    Wasserstein distance between two gaussians.
    """
    mu1 = gauss1[0]
    mu2 = gauss2[0]
    cov1 = gauss1[1]
    cov2 = gauss2[1]
    sigma = cov1 + cov2 - 2 * sqrtm(sqrtm(cov1) @ cov2 @ sqrtm(cov1))
    dist = np.sqrt(np.linalg.norm(mu1 - mu2) ** 2 + np.matrix.trace(sigma))
    return dist


def gaussian_affine_transfo(A, b, x):
    """
    Affine transformation applied to a gaussian x: Tx = Ax + b.
    """
    return (A @ x[0] + b, A @ x[1] @ A.T)


def gaussian_pi_prod(pi, y):
    """
    Matrix vector product between the solution of the optimal transport problem
    and a vector of gaussians.
    """
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


def fit_gaussians(x, Ng):
    """
    Takes a point cloud and fits Ng gaussians on its clusters.
    """
    gm = GaussianMixture(n_components = Ng)
    gm.fit(x)
    moments =  [(gm.means_[i], gm.covariances_[i]) for i in range(Ng)]
    return moments
