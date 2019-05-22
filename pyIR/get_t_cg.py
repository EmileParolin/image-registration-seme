import numpy as np
import scipy
from scipy import optimize
na = np.newaxis



x = np.array([[-1,-1], [-1,1], [1,-1], [6,1]])
y = 1. * x

A = np.random.rand(2,2)
b = np.random.rand(2)
t = np.hstack((A.reshape(4), b))

def get_Ab(t):
    A = t[0:4].reshape(2,2)
    b = t[4:6]
    return A, b

def phi(t, *args):
    x, y = args
    A, b = get_Ab(t)
    r = np.sum(A[na,:,:] * x[:,na,:], axis=1) + b - y
    n = np.linalg.norm(r, axis=1)**2
    return np.sum(r)

def grad_phi(t, *args):
    x, y = args
    A, b = get_Ab(t)
    r = np.sum(A[na,:,:] * x[:,na,:], axis=1) + b - y
    gradA_phi = np.sum(x[:,na,:] * r[:,:,na], axis=0)
    gradb_phi = np.sum(r, axis=0)
    return np.hstack((gradA_phi.reshape(4), gradb_phi))

phi(t,x,y)
grad_phi(t,x,y)

x0 = np.zeros(6)
res = optimize.fmin_cg(phi, x0, fprime=grad_phi, args=(x,y))
