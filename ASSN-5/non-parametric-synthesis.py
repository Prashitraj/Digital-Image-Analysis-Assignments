import numpy as np
import math
from random import randint
import scipy.stats as st
import scipy
import cv2
from time import time


'''generate a 5x5 kernel'''
def generating_kernel(a):
  w_1d = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
  return np.outer(w_1d, w_1d)
 
'''reduce image by 1/2'''
def ireduce(image):
  out = None
  kernel = generating_kernel(0.4)
  outimage = scipy.signal.convolve2d(image,kernel,'same')
  out = outimage[::2,::2]
  return out
 
'''expand image by factor of 2'''
def iexpand(image):
  out = None
  kernel = generating_kernel(0.4)
  outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
  outimage[::2,::2]=image[:,:]
  out = 4*scipy.signal.convolve2d(outimage,kernel,'same')
  return out
 
'''create a gaussain pyramid of a given image'''
def gauss_pyramid(image, levels):
  output = []
  output.append(image)
  tmp = image
  for i in range(0,levels):
    tmp = ireduce(tmp)
    output.append(tmp)
  return output

def process_pixel(x, y, img_data, new_img_data, mask, kernel_size):

    x0 = max(0, x - kernel_size)
    y0 = max(0, y - kernel_size) 
    x1 = min(new_img_data.shape[0] - 1, x + kernel_size)
    y1 = min(new_img_data.shape[1] - 1, y + kernel_size)

    neigh_window = new_img_data[x0 : x1, y0 : y1]

    mask_window = mask[x0 : x1, y0 : y1]
    len_mask = float(len(mask_window==True))

    xs, ys = neigh_window.shape
    img_xsize, img_ysize = img_data.shape

    cx = int(np.floor(xs/2))
    cy = int(np.floor(ys/2))

    candidates = []
    dists = []

    for i in range(xs, img_xsize - xs):
        for j in range(ys, img_ysize - ys):
            if(randint(0,2) != 0): continue
            sub_window = img_data[i : i+xs, j : j+ys]

            # distance
            s = (sub_window - neigh_window)

            summ = s*s*mask_window

            d = np.sum(summ) / len_mask

            candidates.append(sub_window[cx, cy])
            dists.append(d)
    if (len(dists)>0):
        mask = dists - np.min(dists) < 0.2

        candidates = np.extract(mask, candidates)   

        # pick random among candidates
        if len(candidates) < 1:
            return 0.0
        else:
            if len(candidates) != 1:
                r = randint(0, len(candidates) - 1)
            else:
                r = 0

        return candidates[r]
    else: 
        return None


   

def efros(img_data, new_size_x, new_size_y, kernel_size, t):

    patch_size_x, patch_size_y = img.shape
    size_seed_x = size_seed_y = 3

    seed_x = randint(0, size_seed_x)
    seed_y = randint(0, size_seed_y)

    # take 3x3 start image (seed) in the original image
    seed_data = img_data[seed_x : seed_x + size_seed_x, seed_y : seed_y + size_seed_y]

    new_image_data = np.zeros((new_size_x, new_size_y))
    mask = np.ones((new_size_x, new_size_y)) == False

    mask[0: size_seed_x, 0: size_seed_y] = True

    new_image_data[0: size_seed_x, 0: size_seed_y] = seed_data


    # TO DO: non-square images

    it = 0
    for i in range(size_seed_x, new_size_x ):
        print ("Time: ", time() - t, " seconds")
        last_y = size_seed_x + it
        for j in range(0, last_y + 1):

            v = process_pixel(i, j, img_data, new_image_data, mask, kernel_size)

            new_image_data[i, j] = v
            mask[i, j] = True
        for x in range(0, size_seed_y + it + 1):

            v = process_pixel(x, last_y, img_data, new_image_data, mask, kernel_size)

            new_image_data[x, last_y] = v
            mask[x, last_y] = True

        it += 1

    return new_image_data


def gauss_efros (img,newx,newy,ksize,t):
    lg = gauss_pyramid(img,5)
    for i in range(len(lg)):
        lg[i] = efros(lg[i],newx,newy,ksize,t)
    return(lg[len(lg)-1])


img = cv2.imread('tex_sample.png',cv2.IMREAD_GRAYSCALE)
cv2.imwrite('in.jpg',img)
row,col = img.shape
t = time()
out = efros(img, 2*row, col*2, 12, t)
cv2.imwrite('out.png',out)
