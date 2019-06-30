import numpy as np
import scipy
from scipy.optimize import fmin_cg
na = np.newaxis


def get_Ab(t):
    """
    Computes affine transformation attributes (matrix A and vector b)
    from unknown vector fed to optimization routine (coefficients of A and b in
    a unique vector).
    """
    A = t[0:4].reshape(2,2)
    b = t[4:6]
    return A, b


def phi(t, *args):
    """
    Objective function to minimize to obtain the affine transformation.
    """
    # Unpacking data
    mu_1, pi_mu_2, distance, affine_transfo = args
    A, b = get_Ab(t)
    N = len(mu_1)
    assert len(mu_1) == len(pi_mu_2)
    # Computing value of objective function
    r = 0.
    for i in np.arange(N):
        r += distance(affine_transfo(A, b, mu_1[i]), pi_mu_2[i]) ** 2
    return r


def get_initial_guess():
    """
    Initialisation to A = Id and b = 0.
    """
    x0 = np.zeros(6)
    x0[0] = 1
    x0[3] = 1
    return x0


def get_t(mu_1, mu_2, pi, distance, affine_transfo, pi_prod):
    """
    Compute affine transformation.
    """
    args = (mu_1, pi_prod(pi, mu_2), distance, affine_transfo)
    sol = fmin_cg(phi, get_initial_guess(), args=args, full_output=True)
    A, b = get_Ab(sol[0])
    return sol, A, b


def apply_t(A, b, x):
    """
    Apply affine transformation T : x -> Ax + b
    if x are points in R^2
    """
    return np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:]
