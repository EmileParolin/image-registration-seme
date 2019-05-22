import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import cvxopt # Quadratic optimization
from cvxopt import matrix
from cvxopt import solvers
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
    A = coo_matrix((data,(ii, jj)))
    b = np.ones(2*N, dtype=np.int64)
    # Linear programming
    res = linprog(c, A_eq=A.todense(), b_eq=b)
    pi = res.x.reshape(N,N)
    return res, pi


def get_t(x, y, pi):
    """
    Compute linear transformation corresponding for a given optimal
    transformation matrix.
    """
    piy = np.sum(pi[:,:,na] * y[:,na,:], axis=0)
    # matrix M in argmin_x (x'Mx + q'x)
    mis = x[:,na,:] * x[:,:,na]
    Mis = np.kron(np.eye(2), mis)
    M = np.sum(Mis, axis=0)
    # vector q in argmin_x (x'Mx + q'x)
    qis = -2*np.sum(np.tile(x,2) * np.repeat(piy,2,axis=1), axis=0)
    q = -2*np.sum(qis, axis=0)
    # Quadratic optimization
    sol = solvers.qp(2*matrix(M), matrix(q))
    A = np.asarray(sol['x']).reshape(2,2)
    # affine part of transformation
    b = np.mean(y-x, axis=0)
    #b = np.mean(y-np.sum(A[na,:,:] * x[:,na,:], axis=1)*x, axis=0)
    return sol, A, b


N = 25
x = np.vstack((np.linspace(1,N,N), np.linspace(1,N,N))).T
x.shape
y = 4*x[::-1]+2

x = np.array([[-1,-1], [-1,1], [1,-1], [6,1]])
y = 4 * x + 2

res, pi = get_pi(x, y)
sol, A, b = get_t(x, y, pi)
