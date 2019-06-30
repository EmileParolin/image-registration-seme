import numpy as np
import scipy
from scipy.linalg import sqrtm
from sklearn.mixture import GaussianMixture


def fit_gaussians(x, K=50):
    """
    Takes a point cloud and fits K gaussians on its clusters.
    """
    if x.shape[0] < K:
        msg = "Number of points below number of Gaussians K: \n"
        msg += "Either increase the threshold, or reduce K."
        raise(ValueError(msg))
    gm = GaussianMixture(n_components = K)
    gm.fit(x)
    moments = [(gm.means_[i], gm.covariances_[i]) for i in range(K)]
    return moments


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


def gaussian_pi_prod(Pi, x):
    """
    Matrix vector product between the solution of the optimal transport problem
    and a vector of gaussians.
    """
    # Size of vector (and matrix is square of this size)
    N = len(x)
    # Initialisation of result of matrix-vector product
    means = [np.zeros(2) for i in range(N)]
    covariances = [np.zeros((2,2)) for i in range(N)]
    # Matrix-vector product
    for i in range(N):
        for j in range(N):
            means[i] += Pi[i,j] * x[j][0]
            covariances[i] += Pi[i,j] * x[j][1]
    return [(means[i], covariances[i]) for i in range(N)]
