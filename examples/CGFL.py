import matplotlib.pyplot as plt

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pyIR import *

downsampling = "_25" # possible choices "_5", "_10", "_25"
# The smaller the number, the larger the image

# Test case: lung
source = "./inputs/Cas1_lung_biopsy/HLA"+downsampling+".png"
target = "./inputs/Cas1_lung_biopsy/CD8"+downsampling+".png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.7, K=25, use_gaussians=True)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/lung_HLA", use_gaussians=True)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/lung_CD8", use_gaussians=True)
plt.figure(3)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/lung_HLA_CD8")

# Test case: colon
source = "./inputs/Cas2_colon_biopsy/CK7"+downsampling+".png"
target = "./inputs/Cas2_colon_biopsy/HES2_1"+downsampling+".png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.7, K=25, use_gaussians=True)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/colon_CK7", use_gaussians=True)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/colon_HES2_1", use_gaussians=True)
plt.figure(3)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/colon_CK7_HES2_1")

# Test case: breast
source = "./inputs/Cas3_breast_biopsy/RP"+downsampling+".png"
target = "./inputs/Cas3_breast_biopsy/HES_1"+downsampling+".png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.7, K=25, use_gaussians=True)
plt.figure(1)
plot_model(src[3], src[4], "./outputs/breast_RP", use_gaussians=True)
plt.figure(2)
plot_model(tgt[3], tgt[4], "./outputs/breast_HES2_1", use_gaussians=True)
plt.figure(3)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/breast_RP_HES2_1")
