import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.optimize import fmin_cg
from scipy.sparse import coo_matrix
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


N = 50
A = 7*np.eye(2)
b = np.asarray([2., 3.])
x = np.vstack((np.random.normal(0,1,N), np.random.normal(10,4,N))).T
y = np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:]

N = 50
x = np.vstack((np.sort(np.random.normal(0,1,N)), np.sort(np.random.normal(0,10,N)))).T
y = np.vstack((np.sort(np.random.normal(10,10,N)), np.sort(np.random.normal(3,1,N)))).T

N = 50
x = np.vstack((np.random.normal(0,1,N), np.random.normal(0,10,N))).T
y = np.vstack((np.random.normal(10,10,N), np.random.normal(3,5,N))).T

res, pi = get_pi(x, y)
A, b = get_t(x, y, pi)
z = np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:]

plt.scatter(x[:,0], x[:,1])
plt.scatter(y[:,0], y[:,1])
plt.scatter(z[:,0], z[:,1])
