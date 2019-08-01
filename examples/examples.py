import numpy as np
import matplotlib.pyplot as plt

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pyIR import *

# Fake translation to be able to see something on the plot
At = np.eye(2)
bt = np.array([5000, 0])

# Rectangle with rotation, using points, not so good
source = "./inputs/test/rectangle0.png"
target = "./inputs/test/rectangle1.png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.2, K=100, use_gaussians=False)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/rectangle0", use_gaussians=False)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/rectangle1", use_gaussians=False)
plt.figure(3)
plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
plt.figure(4)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/rec_points")

# Rectangle with rotation, works well most of the times with 2 gaussians
source = "./inputs/test/rectangle0.png"
target = "./inputs/test/rectangle1.png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.2, K=2, use_gaussians=True)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/rectangle0", use_gaussians=True)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/rectangle1", use_gaussians=True)
plt.figure(3)
plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
plt.figure(4)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/rec_gaussians")

# Malevich rotation
source = "./inputs/test/malevich0.png"
target = "./inputs/test/malevich1.png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.2, K=2, use_gaussians=True)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/malevich0", use_gaussians=True)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/malevich1", use_gaussians=True)
plt.figure(3)
plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
plt.figure(4)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/malevitch_rotation")

# Malevich translation
source = "./inputs/test/malevich0.png"
target = "./inputs/test/malevich2.png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.2, K=2, use_gaussians=True)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/malevich0", use_gaussians=True)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/malevich2", use_gaussians=True)
plt.figure(3)
plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
plt.figure(4)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/malevitch_translation")

# Malevich dilatation
source = "./inputs/test/malevich0.png"
target = "./inputs/test/malevich3.png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.2, K=2, use_gaussians=True)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/malevich0", use_gaussians=True)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/malevich3", use_gaussians=True)
plt.figure(3)
plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
plt.figure(4)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/malevitch_dilatation")

# Pacman, does not works well
source = "./inputs/test/pacman.png"
target = "./inputs/test/pacman0.png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.2, K=25, use_gaussians=True)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/pacman", use_gaussians=True)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/pacman0", use_gaussians=True)
plt.figure(3)
plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
plt.figure(4)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/pacman_flip")

# Pacman, does not works well
source = "./inputs/test/pacman0.png"
target = "./inputs/test/pacman3.png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.2, K=25, use_gaussians=True)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/pacman0", use_gaussians=True)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/pacman3", use_gaussians=True)
plt.figure(3)
plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
plt.figure(4)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/pacman_rot")

# Small cat, translation, small rotation, reduction: not bad
source = "./inputs/test/cat0.png"
target = "./inputs/test/cat4.png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.2, K=10, use_gaussians=True)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/cat0", use_gaussians=True)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/cat4", use_gaussians=True)
plt.figure(3)
plot_images(At @ A, b+bt, src[0], apply_t(At, bt, tgt[0]))
plt.figure(4)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/cat")
