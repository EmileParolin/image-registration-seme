import numpy as np
import matplotlib.pyplot as plt

from .image import get_image
from .image import points_above_threshold

from .points import euclidian_distance
from .points import euclidian_affine_transfo
from .points import points_pi_prod

from .gaussians import wasserstein_distance
from .gaussians import gaussian_affine_transfo
from .gaussians import gaussian_pi_prod
from .gaussians import fit_gaussians

from .transport import get_pi

from .projection import get_t
from .projection import apply_t


def get_transfo(mu_1, mu_2, distance, affine_transfo, pi_prod):
    """
    Main part of the algorithm, solves optimal transport problem and performs
    projection step.

    Inputs are two models.
    """
    # Solve optimal transport problem
    res, pi = get_pi(mu_1, mu_2, distance)
    # Projection: Compute affine transformation
    sol, A, b = get_t(mu_1, mu_2, pi, distance, affine_transfo, pi_prod)
    return A, b, res, sol, pi


def image_registration(source, target, threshold=0.2, K=50, use_gaussians=True):
    """
    Main routine to perform image registration.

    Inputs (source, target) are two file names.
    Output features the affine transformation computed in the form of a matrix
    A and a vector b (first two outputs) and several other info.
    """
    # Get correct routines depending on modelling type
    if use_gaussians :
        get_model = fit_gaussians
        distance = wasserstein_distance
        affine_transfo = gaussian_affine_transfo
        pi_prod = gaussian_pi_prod
    else :
        get_model = uniform_sampling
        distance = euclidian_distance
        affine_transfo = euclidian_affine_transfo
        pi_prod = points_pi_prod
    # Image data
    ij_1, xBW_1, xRGB_1 = get_image(source)
    ij_2, xBW_2, xRGB_2 = get_image(target)
    # Positions of image pixels with intensity above threshold
    ij_black_1 = points_above_threshold(ij_1, xBW_1, threshold=threshold)
    ij_black_2 = points_above_threshold(ij_2, xBW_2, threshold=threshold)
    # Models
    mu_1 = get_model(ij_black_1, K=K)
    mu_2 = get_model(ij_black_2, K=K)
    # Get T (Transport + Projection)
    A, b, res, sol, pi = get_transfo(mu_1, mu_2, distance, affine_transfo, pi_prod)
    # Preparation of output
    src = ij_1, xBW_1, xRGB_1, ij_black_1, mu_1
    tgt = ij_2, xBW_2, xRGB_2, ij_black_2, mu_2
    return A, b, res, sol, pi, src, tgt


def plot_images(A, b, ij_1, ij_2):
    """
    Utility function to compute the image of the source by the affine
    transformation and quickly plot the 3 images on a single figure.
    """
    # Image reconstruction using affine transformation
    ij_3 = apply_t(A, b, ij_1)
    plt.cla()
    plt.scatter(ij_1[:,0], ij_1[:, 1], marker=".", label="mu_1")
    plt.scatter(ij_2[:,0], ij_2[:, 1], marker="x", label="mu_2")
    plt.scatter(ij_3[:,0], ij_3[:, 1], marker="+", label="T mu_1")
    plt.legend()


def save_colored_images(A, b, ij_1, ij_2, xRGB_1, xRGB_2, name):
    """
    Compute the image of the source by the affine transformation and save the 3
    images in 3 different files.
    """
    # Image reconstruction using affine transformation
    ij_3 = apply_t(A, b, ij_1)
    # Image limits
    ij_x = np.hstack((ij_1[:,0], ij_2[:,0], ij_3[:,0]))
    ij_y = np.hstack((ij_1[:,1], ij_2[:,1], ij_3[:,1]))
    xmin = np.min(ij_x)
    xmax = np.max(ij_x)
    ymin = np.min(ij_y)
    ymax = np.max(ij_y)
    # Figure
    for (ij, xRGB, postfix) in zip([ij_1, ij_2, ij_3],
                                   [xRGB_1, xRGB_2, xRGB_1],
                                   ["_source", "_target", "_reconstructed"]):
        plt.cla()
        plt.scatter(ij[:,0], ij[:,1], c=xRGB/255.0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(name+postfix)
