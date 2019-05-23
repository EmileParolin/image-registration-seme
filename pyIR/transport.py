import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.optimize import fmin_cg
from scipy.sparse import coo_matrix
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
na = np.newaxis


def get_pi(x, y):
    """
    Get matrix of optimal transport problem.
    """
    # Number of points
    N = x.shape[0]
    # Cost vector
    c = (np.linalg.norm(x[:,:,na]-y.T[na,:,:], axis=1).reshape(N*N))**2
    # 2 * N * N constraints
    ln = np.linspace(0,N-1,N, dtype=np.int64)
    ii_1 = np.repeat(ln, N).reshape(N*N)
    jj_1 = np.linspace(0,N*N-1,N*N, dtype=np.int64)
    ii_2 = ii_1 + N
    jj_2 = np.tile(N*ln.T, N) + ii_1
    ii = np.hstack((ii_1, ii_2))
    jj = np.hstack((jj_1, jj_2))
    data = np.ones(2*N*N)
    A = (coo_matrix((data,(ii, jj)))).todense()[0:-1,:]
    b = np.ones(2*N, dtype=np.int64)[0:-1]
    # Linear programming
    res = linprog(c, A_eq=A, b_eq=b)
    pi = res.x.reshape(N,N)
    return res, pi


def get_Ab(t):
    A = t[0:4].reshape(2,2)
    b = t[4:6]
    return A, b


def phi(t, *args):
    x, y = args
    A, b = get_Ab(t)
    r = np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:] - y
    n = np.linalg.norm(r, axis=1)**2
    return np.sum(n)


def get_t(x, y, pi):
    piy = np.sum(pi[:,:,na] * y[na,:,:], axis=1)
    x0 = np.zeros(6)
    x0[0] = 1
    x0[3] = 1
    A, b = get_Ab(fmin_cg(phi, x0, args=(x, piy)))
    return A, b


def wasserstein(gauss1, gauss2):
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
