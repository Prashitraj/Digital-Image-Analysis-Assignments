import sys
import cv2
import random
import numpy as np
import math


def gaussian_noise(image):
    gauss = np.random.normal(0,20, np.shape(img))
    noisy_img = img + gauss
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img

def anisotropic(img,filters):
    for r in range(15):
    # approximate gradients
        gradients = [ cv2.filter2D(img, -1,w) for w in filters ]
    # approximate diffusion function
        diff = [ 1./(1 + (c/16)**2) for c in gradients]
    # update image
        terms = [diff[i]*gradients[i] for i in range(4)]
        terms += [0.5*diff[i]*gradients[i] for i in range(4, 8)]
        img = img + 0.06*(sum(terms))
    return (img)



filters = [
    np.array(
            [[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64
    ),
    np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64
    ),
]


infile = sys.argv[1]
midfile = sys.argv[2]
outfile = sys.argv[3]
img = cv2.imread(infile,cv2.IMREAD_GRAYSCALE)
noisy = gaussian_noise(img)
cv2.imwrite(midfile,noisy)
outimg = anisotropic(noisy,filters)
cv2.imwrite(outfile,outimg)