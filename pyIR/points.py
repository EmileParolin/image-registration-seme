import numpy as np
na = np.newaxis


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
