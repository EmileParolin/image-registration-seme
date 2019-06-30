import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
na = np.newaxis


def get_pi(mu_1, mu_2, distance):
    """
    Get matrix of optimal transport problem.
    """
    # Number of points
    N = len(mu_1)
    assert len(mu_1) == len(mu_2)
    # Cost matrix
    c = np.zeros((N,N), dtype=np.float64)
    for i in np.arange(N):
        for j in np.arange(N):
            c[i,j] = distance(mu_1[i], mu_2[j]) ** 2
    c = c.reshape(N*N)
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
