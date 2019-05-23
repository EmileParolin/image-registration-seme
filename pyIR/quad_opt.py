import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import cvxopt # Quadratic optimization
from cvxopt import matrix
from cvxopt import solvers
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
    piy = 1.*y#np.dot(pi[:,:,na], y.T).T[:,:,0]
    # matrix M in argmin_x (x'Mx + q'x)
    mais = x[:,na,:] * x[:,:,na]
    Mais = np.kron(np.eye(2), mais)
    Ma = np.sum(Mais, axis=0)
    sx = np.sum(x, axis=0)
    mb = np.vstack((np.kron(sx, [1,0]).reshape(2,2), np.kron(sx, [0,1]).reshape(2,2)))
    M = np.vstack((np.hstack((Ma, mb)), np.hstack((mb.T, np.eye(2)))))
    print(np.linalg.det(M))
    print(np.linalg.eigvals(M))
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
A = 7*np.eye(2)
b = np.asarray([2., 3.])
x = np.vstack((np.random.normal(0,1,N), np.random.normal(10,4,N))).T
y = np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:]

#res, pi = get_pi(x, y)
pi = np.eye(y.shape[0])
sol, A, b = get_t(x, y, pi)

z = np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:]

plt.scatter(x[:,0], x[:,1])
plt.scatter(y[:,0], y[:,1])
plt.scatter(z[:,0], z[:,1])
