import numpy as np
from get_image import getImage

from transport import get_pi
from transport import get_t

from points import euclidian_distance
from points import euclidian_affine_transfo
from points import points_pi_prod

from gaussians import wasserstein_distance
from gaussians import gaussian_affine_transfo
from gaussians import gaussian_pi_prod
from gaussians import fit_gaussians

import matplotlib.pyplot as plt
na = np.newaxis


def reconstruct(A, b, x):
    return np.dot(A[na,:,:], x.T).T[:,:,0] + b[na,:]


def reconstruct_colored_image(A, b, x, y, XRGB, YRGB, name):
    z = reconstruct(A, b, x)
    # Rotation
    R = np.array([[0,1],[-1,0]])
    x = reconstruct(R, 0*b, x)
    y = reconstruct(R, 0*b, y)
    z = reconstruct(R, 0*b, z)
    # Limits
    xmin = np.min(np.array([np.min(x[:,0]), np.min(y[:,0]), np.min(z[:,0])]))
    xmax = np.max(np.array([np.max(x[:,0]), np.max(y[:,0]), np.max(z[:,0])]))
    ymin = np.min(np.array([np.min(x[:,1]), np.min(y[:,1]), np.min(z[:,1])]))
    ymax = np.max(np.array([np.max(x[:,1]), np.max(y[:,1]), np.max(z[:,1])]))
    # Source
    plt.cla()
    plt.scatter(x[:,0], x[:,1], c=XRGB/255.0)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(name+"_source")
    # Target
    plt.cla()
    plt.scatter(y[:,0], y[:,1], c=YRGB/255.0)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(name+"_target")
    # Reconstructed
    plt.cla()
    plt.scatter(z[:,0], z[:,1], c=XRGB/255.0)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(name+"_reconstructed")


def image_registration(source, target, n_s, threshold, Ng, use_points):

    # Image models
    Ximg, Ximg_black, Ximg_s = getImage(source, n_s=n_s, threshold=threshold)
    Yimg, Yimg_black, Yimg_s = getImage(target, n_s=n_s, threshold=threshold)
    # Position and color of all image pixels
    xij, XBW, XRGB = Ximg
    yij, YBW, YRGB = Yimg
    # Position and color of image pixels above threshold
    xij_black, X_black = Ximg_black
    yij_black, Y_black = Yimg_black
    # Position and color of image pixels above threshold sample subset
    xij_s, X_s = Ximg_s
    yij_s, Y_s = Yimg_s
    # Translation of target image
    Xmax = np.max(xij[:,0])
    yij[:,0] += Xmax
    yij_black[:,0] += Xmax
    yij_s[:,0] += Xmax
    if use_points :
        # Position and color of sample image pixels
        x = xij_s
        y = yij_s
        # routines for points
        distance = euclidian_distance
        affine_transfo = euclidian_affine_transfo
        pi_prod = points_pi_prod
    else :
        # Gaussian fits
        x = fit_gaussians(xij_black, Ng)
        y = fit_gaussians(yij_black, Ng)
        # routines for gaussians
        distance = wasserstein_distance
        affine_transfo = gaussian_affine_transfo
        pi_prod = gaussian_pi_prod
    # Solve optimal transport problem
    res, pi = get_pi(x, y, distance)
    # Compute affine transformation
    sol, A, b = get_t(x, y, pi, distance, affine_transfo, pi_prod)
    # Apply affine transformation to image pixels
    zij_black = reconstruct(A, b, xij_black)
    # Rotation
    R = np.array([[0,1],[-1,0]])
    xij_black = reconstruct(R, 0*b, xij_black)
    yij_black = reconstruct(R, 0*b, yij_black)
    zij_black = reconstruct(R, 0*b, zij_black)
    # Plot
    plt.cla()
    plt.scatter(xij_black[:,0], xij_black[:,1], marker=".", label="source")
    plt.scatter(yij_black[:,0], yij_black[:,1], marker="x", label="target")
    plt.scatter(zij_black[:,0], zij_black[:,1], marker="+", label="reconstructed")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    return res, sol, pi, A, b, (xij, yij, xij_black, yij_black, xij_s, yij_s, XBW, XRGB, YBW, YRGB)


if False :

    # Rectangle with rotation, using points, not so good
    source = "../images/test/rectangle0.png"
    target = "../images/test/rectangle1.png"
    res, sol, pi, A, b, x = image_registration(source, target, 50, 0.2, 2, True)
    #reconstruct_colored_image(A, b, x[0], x[1], x[7], x[9], "rec")

    # Rectangle with rotation, works well most of the times with 2 gaussians
    source = "../images/test/rectangle0.png"
    target = "../images/test/rectangle1.png"
    res, sol, pi, A, b, x = image_registration(source, target, 50, 0.2, 2, False)

    # Malevich translation
    source = "../images/test/malevich0.png"
    target = "../images/test/malevich2.png"
    res, sol, pi, A, b, x = image_registration(source, target, 50, 0.2, 2, False)

    # Malevich dilatation
    source = "../images/test/malevich0.png"
    target = "../images/test/malevich3.png"
    res, sol, pi, A, b, x = image_registration(source, target, 50, 0.2, 2, False)
    reconstruct_colored_image(A, b, x[0], x[1], x[7], x[9], "malevich")

    # Malevich rotation
    source = "../images/test/malevich0.png"
    target = "../images/test/malevich1.png"
    res, sol, pi, A, b, x = image_registration(source, target, 50, 0.2, 5, False)

    # Pacman, does not works well
    source = "../images/test/pacman0.png"
    target = "../images/test/pacman3.png"
    res, sol, pi, A, b, x = image_registration(source, target, 50, 0.2, 10, False)

    # Small cat, translation, small rotation, reduction: not bad
    source = "../images/test/cat0.png"
    target = "../images/test/cat4.png"
    res, sol, pi, A, b, x = image_registration(source, target, 50, 0.2, 50, False)
    reconstruct_colored_image(A, b, x[0], x[1], x[7], x[9], "cat")

    # Test case: lung
    source = "../images/Cas1_lung_biopsy/HLA.png"
    target = "../images/Cas1_lung_biopsy/cd.png"
    res, sol, pi, A, b, x = image_registration(source, target, 50, 0.2, 25, False)
    reconstruct_colored_image(A, b, x[0], x[1], x[7], x[9], "lung")

    # Test case: colon
    source = "../images/Cas2_colon_biopsy/colon_CK7.png"
    target = "../images/Cas2_colon_biopsy/colon_HES2.png"
    res, sol, pi, A, b, x = image_registration(source, target, 50, 0.2, 25, False)
    reconstruct_colored_image(A, b, x[0], x[1], x[7], x[9], "colon")

    # Test case: breast
    source = "../images/Cas3_breast_biopsy/RP.png"
    target = "../images/Cas3_breast_biopsy/HES.png"
    res, sol, pi, A, b, x = image_registration(source, target, 50, 0.2, 25, False)
    reconstruct_colored_image(A, b, x[0], x[1], x[7], x[9], "breast")
