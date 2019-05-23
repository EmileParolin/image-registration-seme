import numpy as np
import scipy
import cvxopt # Quadratic optimization
from cvxopt import matrix
from cvxopt import solvers
na = np.newaxis

N = 50
x = np.vstack((np.sort(np.random.normal(0,1,N)), np.sort(np.random.normal(0,1,N)))).T
y = np.vstack((np.sort(np.random.normal(5,1,N)), np.sort(np.random.normal(5,1,N)))).T

x = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])

x = np.array([[-1,-1], [1,1], [2,3], [6,1]])

epsilon = 1.
x = np.array([[1,2], [2,3]])
y = 5. * x + 7.

N = 50
x = np.vstack((np.sort(np.random.normal(0,1,N)), np.sort(np.random.normal(0,1,N)))).T
y = np.vstack((np.sort(np.random.normal(5,1,N)), np.sort(np.random.normal(5,1,N)))).T
pi = np.eye(y.shape[0])
piy = np.sum(pi[:,:,na] * y[:,na,:], axis=0)
mais = x[:,na,:] * x[:,:,na]
Mais = np.kron(np.eye(2), mais)
Ma = np.sum(Mais, axis=0)
sx = np.sum(x, axis=0)
mb = np.vstack((np.kron(sx, [1,0]).reshape(2,2), np.kron(sx, [0,1]).reshape(2,2)))
M = np.vstack((np.hstack((Ma, mb)), np.hstack((mb.T, np.eye(2)))))
np.linalg.eigvals(M)
w, v = np.linalg.eig(M)
np.linalg.det(M)
M
qais = np.tile(x,2) * np.repeat(piy,2,axis=1)
qa = np.sum(qais, axis=0)
qb = np.sum(piy, axis=0)
q = -2 * np.hstack((qa, qb))
sol = solvers.qp(matrix(2 * M), matrix(q))
Ab = np.asarray(sol['x'])
A = Ab[0:4].reshape(2,2)
b = Ab[4:6]
np.round(A)
np.round(b)

M += epsilon * np.eye(6)
np.linalg.eigvals(M)
