import numpy as np
na = np.newaxis


def uniform_sampling(x, K = 250):
    """
    Uniform sampling of K points among cloud of points x.
    """
    # Number of points
    N = x.shape[0]
    # Sampling without replacement
    sampled = np.random.choice(np.arange(1,N), size=min(K, N-1), replace=False)
    # Matrix of sampled black coordinates
    return x[sampled,:]


def euclidian_distance(x, y):
    """
    Euclidian norm between two points.
    """
    return np.linalg.norm(x - y)


def euclidian_affine_transfo(A, b, x):
    """
    Euclidian affine transformation of a point x: Tx = Ax + b.
    """
    return np.dot(A, x.T).T + b


def points_pi_prod(pi, y):
    """
    Matrix vector product between the solution of the optimal transport problem
    and a vector of points.
    """
    return np.sum(pi[:,:,na] * y[na,:,:], axis=1)
