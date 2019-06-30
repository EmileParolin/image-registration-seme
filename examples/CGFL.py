from pyIR import *

downsampling = "_25" # possible choices "_5", "_10", "_25"

# Test case: lung
source = "./inputs/Cas1_lung_biopsy/HLA"+downsampling+".png"
target = "./inputs/Cas1_lung_biopsy/CD8"+downsampling+".png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.3, K=25, use_gaussians=True)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/lung_HLA_CD8")

# Test case: colon
source = "./inputs/Cas2_colon_biopsy/CK7"+downsampling+".png"
target = "./inputs/Cas2_colon_biopsy/HES2_1"+downsampling+".png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.3, K=25, use_gaussians=True)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/colon_CK7_HES2_1")

# Test case: breast
source = "./inputs/Cas3_breast_biopsy/RP"+downsampling+".png"
target = "./inputs/Cas3_breast_biopsy/HES_1"+downsampling+".png"
A, b, res, sol, pi, src, tgt = image_registration(source, target,
        threshold=0.3, K=25, use_gaussians=True)
save_colored_images(A, b, src[0], tgt[0], src[2], tgt[2], "./outputs/breast_RP_HES2_1")
