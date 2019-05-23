import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.optimize import fmin_cg
from scipy.sparse import coo_matrix
na = np.newaxis


def get_pi(x, y, distance):
    """
    Get matrix of optimal transport problem.
    """
    # Number of points
    N = len(x)
    assert len(x) == len(y)
    # Cost matrix
    c = np.zeros((N,N), dtype=np.float64)
    for i in np.arange(N):
        for j in np.arange(N):
            c[i,j] = distance(x[i], y[j]) ** 2
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


def phi(t, *args):
    """
    Objective function to minimize to obtain the linear map.
    """
    x, y, distance, affine_transfo = args
    N = len(x)
    assert len(x) == len(y)
    A, b = get_Ab(t)
    r = 0.
    for i in np.arange(N):
        r += distance(affine_transfo(A, b, x[i]), y[i]) ** 2
    return r


def get_Ab(t):
    """
    Computes linear transformation attributes (namely matrix A and vector b)
    from unknown vector fed to optimization routine (coefficients of A and b in
    a unique vector).
    """
    A = t[0:4].reshape(2,2)
    b = t[4:6]

    #A = t[0] * np.array([[np.cos(t[1]), -np.sin(t[1])], [np.sin(t[1]), np.cos(t[1])]])
    #b = t[2:4]

    #A = t[0] * np.array([[np.cos(t[1]), -np.sin(t[1])], [np.sin(t[1]), np.cos(t[1])]])
    #b = np.zeros(2)
    return A, b


def get_initial_guess():
    x0 = np.zeros(6)
    x0[0] = 1
    x0[3] = 1

    #x0 = np.zeros(4)
    #x0[0] = 1

    #x0 = np.zeros(2)
    #x0[0] = 1
    return x0


def get_t(x, y, pi, distance, affine_transfo, pi_prod):
    """
    Compute linear transformation
    """
    piy = pi_prod(pi, y)
    x0 = get_initial_guess()
    args = (x, piy, distance, affine_transfo)
    A, b = get_Ab(fmin_cg(phi, x0, args=args))
    return A, b
