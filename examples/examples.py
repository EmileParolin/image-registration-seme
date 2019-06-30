import numpy as np

from pyIR import *

# Fake translation to be able to see something on the plot
At = np.eye(2)
bt = np.array([5000, 0])

if False :

    # Rectangle with rotation, using points, not so good
    source = "./inputs/test/rectangle0.png"
    target = "./inputs/test/rectangle1.png"
    A, b, res, sol, pi, src, tgt = image_registration(source, target,
            threshold=0.2, K=100, use_gaussians=False)
    plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
    save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/rec_points")

    # Rectangle with rotation, works well most of the times with 2 gaussians
    source = "./inputs/test/rectangle0.png"
    target = "./inputs/test/rectangle1.png"
    A, b, res, sol, pi, src, tgt = image_registration(source, target,
            threshold=0.2, K=2, use_gaussians=True)
    plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
    save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/rec_gaussians")

    # Malevich translation
    source = "./inputs/test/malevich0.png"
    target = "./inputs/test/malevich2.png"
    A, b, res, sol, pi, src, tgt = image_registration(source, target,
            threshold=0.2, K=2, use_gaussians=True)
    plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
    save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/malevitch_translation")

    # Malevich dilatation
    source = "./inputs/test/malevich0.png"
    target = "./inputs/test/malevich3.png"
    A, b, res, sol, pi, src, tgt = image_registration(source, target,
            threshold=0.2, K=2, use_gaussians=True)
    plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
    save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/malevitch_dilatation")

    # Malevich rotation
    source = "./inputs/test/malevich0.png"
    target = "./inputs/test/malevich1.png"
    A, b, res, sol, pi, src, tgt = image_registration(source, target,
            threshold=0.2, K=2, use_gaussians=True)
    plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
    save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/malevitch_rotation")

    # Pacman, does not works well
    source = "./inputs/test/pacman0.png"
    target = "./inputs/test/pacman3.png"
    A, b, res, sol, pi, src, tgt = image_registration(source, target,
            threshold=0.2, K=10, use_gaussians=True)
    plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
    save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/pacman")

    # Small cat, translation, small rotation, reduction: not bad
    source = "./inputs/test/cat0.png"
    target = "./inputs/test/cat4.png"
    A, b, res, sol, pi, src, tgt = image_registration(source, target,
            threshold=0.2, K=10, use_gaussians=True)
    plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
    save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/cat")
