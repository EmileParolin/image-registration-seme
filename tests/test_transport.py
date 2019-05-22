import numpy as np
from pyIR import get_pi
from pyIR import get_t
na = np.newaxis

# Data points: random
N = 25
x = np.random.rand(N,2)
y = np.random.rand(N,2)
pi = get_pi(x, y)

# Data points: linear transform
A = np.random.rand(2,2)
b = np.random.rand(2)
N = 4
x = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
y = np.sum(A[na,:,:] * x[:,na,:], axis=1)+b
pi = get_pi(x, y)

y = np.copy(x)
y[2,:] = 0.5*(y[1,:]+y[2,:])
pi = get_pi(x, y)

# Data points: square
x = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
y = 4.*x + 2.
pi = get_pi(x, y)
A,b = get_t(x, y, pi)
