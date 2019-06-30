import numpy as np

from pyIR import *

# Random affine transformation
A0 = 3 * np.random.rand(2,2) + 7 * np.eye(2)
b0 = 8 * np.random.rand(2)

# Fake source points
N = 25
xa = np.vstack((np.random.normal(18,1,N), np.random.normal(18,10,N))).T
xb = np.vstack((np.random.normal(29,9,N), np.random.normal(3,0.5,N))).T
x = np.vstack((xa, xb))
# Fake target points
y = np.array([euclidian_affine_transfo(A0, b0, x[i,:]) for i in range(x.shape[0])])
# Transfo
A, b, res, sol, pi = get_transfo(x, y, euclidian_distance,
        euclidian_affine_transfo, points_pi_prod)
z = apply_t(A, b, x)
# Error
err = np.linalg.norm(z - y)
print(err)
# Plot
plot_images(A, b, x, y)

# Fake source Gaussian mixture
N = 5
ma = np.vstack((np.random.normal(18,1,N), np.random.normal(18,10,N))).T
mb = np.vstack((np.random.normal(29,9,N), np.random.normal(3,0.5,N))).T
mx = np.vstack((xa, xb))
Sx = [5*np.random.rand(2,2) for i in range(2*N)]
Sx = [np.dot(s, s.T) for s in Sx] # symmetric positive semi-definite covariance matrices
x = [(mx[i,:],Sx[i]) for i in range(2*N)]
# Fake target Gaussian mixture
y = [gaussian_affine_transfo(A0, b0, x[i]) for i in range(len(x))]
my = apply_t(A0, b0, mx)
# Transfo
A, b, res, sol, pi = get_transfo(x, y, wasserstein_distance,
        gaussian_affine_transfo, gaussian_pi_prod)
mz = apply_t(A, b, mx)
# Error
err = np.linalg.norm(mz - my)
print(err)
# Plot
plot_images(A, b, mx, my)
