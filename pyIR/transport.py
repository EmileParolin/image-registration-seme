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
    mais = x[:,na,:] * x[:,:,na]
    Mais = np.kron(np.eye(2), mais)
    Ma = np.sum(Mais, axis=0)
    sx = np.sum(x, axis=0)
    mb = np.vstack((np.kron(sx, [1,0]).reshape(2,2), np.kron(sx, [0,1]).reshape(2,2)))
    M = np.vstack((np.hstack((Ma, mb)), np.hstack((mb.T, np.eye(2)))))
    for i in np.arange(4):
        M[i,i] += 1.
    # vector q in argmin_x (x'Mx + q'x)
    qais = np.tile(x,2) * np.repeat(piy,2,axis=1)
    qa = np.sum(qais, axis=0)
    qb = np.sum(piy, axis=0)
    q = -2 * np.hstack((qa, qb))
    # Quadratic optimization: expect P, q in argmin_x (0.5 * x'Px + q'x)
    sol = solvers.qp(matrix(2 * M), matrix(q))
    Ab = np.asarray(sol['x'])
    A = Ab[0:4].reshape(2,2)
    b = Ab[4:6]
    return sol, A, b



N = 25
x = np.vstack((np.linspace(1,N,N), np.linspace(1,N,N))).T
x.shape
y = 4. * x[::-1] + 2.

x = np.array([[1,2]])
y = np.array([[3,4]])
pi = np.eye(1)
y = 4. * x + 2.
y = 4. * x + 2.

N = 4
x = np.vstack((np.random.normal(0,1,N), np.zeros(N))).T
y = np.vstack((np.random.normal(0,10,N), np.zeros(N))).T
y = 2*x
pi = np.eye(N)

x = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
x

x = np.array([[1,-1]])
y = np.array([[1,0]])

x = np.vstack((np.linspace(1,N,N), -3*np.linspace(1,N,N)+4)).T
x

N = 50
x = np.vstack((np.sort(np.random.normal(0,1,N)), np.zeros(N))).T
y = np.vstack((np.sort(np.random.normal(5,1,N)), np.zeros(N))).T
res, pi = get_pi(x, y)
sol, A, b = get_t(x, y, pi)
np.round(A)
np.round(b)

y = 1. * x
res, pi = get_pi(x, y)
sol, A, b = get_t(x, y, pi)
np.round(A)
np.round(b)

y = 2. * x
res, pi = get_pi(x, y)
sol, A, b = get_t(x, y, pi)
np.round(A)
np.round(b)

y = 1. * x + 3.
res, pi = get_pi(x, y)
sol, A, b = get_t(x, y, pi)
np.round(A)
np.round(b)

y = 4. * x + 3.
res, pi = get_pi(x, y)
sol, A, b = get_t(x, y, pi)
np.round(A)
np.round(b)

####################################

x = np.array([[-1,-1], [-1,1], [1,-1], [6,1]])
y = 1. * x
res, pi = get_pi(x, y)
sol, A, b = get_t(x, y, pi)
np.round(A)
np.round(b)

piy = 1*y#np.sum(pi[:,:,na] * y[:,na,:], axis=0)
# matrix M in argmin_x (x'Mx + q'x)
mais = x[:,na,:] * x[:,:,na]
Mais = np.kron(np.eye(2), mais)
Ma = np.sum(Mais, axis=0)
sx = np.sum(x, axis=0)
mb = np.vstack((np.kron(sx, [1,0]).reshape(2,2), np.kron(sx, [0,1]).reshape(2,2)))
M = np.vstack((np.hstack((Ma, mb)), np.hstack((mb.T, np.eye(2)))))
M += np.eye(6)
# vector q in argmin_x (x'Mx + q'x)
qais = np.tile(x,2) * np.repeat(piy,2,axis=1)
qa = np.sum(qais, axis=0)
qb = np.sum(piy, axis=0)
q = -2 * np.hstack((qa, qb))
# Quadratic optimization: expect P, q in argmin_x (0.5 * x'Px + q'x)
sol = solvers.qp(matrix(2 * M), matrix(q))
Ab = np.asarray(sol['x'])
A = Ab[0:4].reshape(2,2)
b = Ab[4:6]
