"""
Image normalization for DDPM. Images loaded via torchvision typically live
in [0, 1]. DDPM's forward process adds zero-centered Gaussian noise, so
images are remapped to [-1, 1] to keep the whole process symmetric around
0 — matching the noise distribution's own centering.
"""


def normalize_to_neg_one_to_one(x):
    """[0, 1] -> [-1, 1]"""
    return x * 2 - 1


def unnormalize_to_zero_to_one(x):
    """[-1, 1] -> [0, 1] (inverse of the above, needed to actually view
    generated images with matplotlib/PIL)"""
    return (x + 1) / 2