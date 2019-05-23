import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
na = np.newaxis



N = 500
x = np.vstack((np.sort(np.random.normal(0,1,N)), np.sort(np.random.normal(10,4,N)))).T
x = np.vstack((np.random.normal(0,1,N), np.random.normal(10,4,N))).T

#y = np.vstack((np.sort(np.random.normal(5,1,N)), np.sort(np.random.normal(5,1,N)))).T

A = 7*np.eye(2)
b = np.asarray([2., 3.])
y = np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:]


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

def grad_phi(t, *args):
    x, y = args
    A, b = get_Ab(t)
    r = np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:] - y
    gradA_phi = np.sum(x[:,na,:] * r[:,:,na], axis=0)
    gradb_phi = np.sum(r, axis=0)
    return np.hstack((gradA_phi.reshape(4), gradb_phi))

A = np.random.rand(2,2)
b = np.random.rand(2)
t = np.hstack((A.reshape(4), b))
phi(t,x,y)
#grad_phi(t,x,y)

x0 = np.zeros(6)
x0[0] = 1
x0[3] = 1

A, b = get_Ab(optimize.fmin_cg(phi, x0, fprime=grad_phi, args=(x,y)))

A, b = get_Ab(optimize.fmin_cg(phi, x0, args=(x,y)))
z = np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:]

A
b

plt.scatter(x[:,0], x[:,1])
plt.scatter(y[:,0], y[:,1])
plt.scatter(z[:,0], z[:,1])
