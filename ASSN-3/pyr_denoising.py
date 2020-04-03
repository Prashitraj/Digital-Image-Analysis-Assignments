import sys
import os
import numpy as np
import cv2
import scipy
from scipy.stats import norm
from scipy.signal import convolve2d
import math


def gaussian_noise(img):
    gauss = np.random.normal(0, 20, np.shape(img))
    noisy_img = img + gauss
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img

'''split rgb image to its channels'''
def split_rgb(image):
  red = None
  green = None
  blue = None
  (blue, green, red) = cv2.split(image)
  return [blue,green,red]
 
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
 
'''build a laplacian pyramid'''
def lapl_pyramid(gauss_pyr):
  output = []
  k = len(gauss_pyr)
  for i in range(0,k-1):
    gu = gauss_pyr[i]
    egu = iexpand(gauss_pyr[i+1])
    if egu.shape[0] > gu.shape[0]:
       egu = np.delete(egu,(-1),axis=0)
    if egu.shape[1] > gu.shape[1]:
      egu = np.delete(egu,(-1),axis=1)
    output.append(gu - egu)
  output.append(gauss_pyr.pop())
  return output

'''Reconstruct the image based on its laplacian pyramid.'''
def collapse(lapl_pyr):
  output = None
  output = np.zeros((lapl_pyr[0].shape[0],lapl_pyr[0].shape[1]), dtype=np.float64)
  for i in range(len(lapl_pyr)-1,0,-1):
    lap = iexpand(lapl_pyr[i])
    lapb = lapl_pyr[i-1]
    if lap.shape[0] > lapb.shape[0]:
      lap = np.delete(lap,(-1),axis=0)
    if lap.shape[1] > lapb.shape[1]:
      lap = np.delete(lap,(-1),axis=1)
    tmp = lap + lapb
    lapl_pyr.pop()
    lapl_pyr.pop()
    lapl_pyr.append(tmp)
    output = tmp
  return output

def hard_thresholding (list,t):
  for i in range (2):
    row,col = list[i].shape
    for x in range(row):
      for y in range(col):
        if (abs(list[i][x][y])<t):
          list[i][x][y] = 0
  return list

def soft_thresholding (list,t):
  for i in range (2):
    row,col = list[i].shape
    for x in range(row):
      for y in range(col):
        if (list[i][x][y]>t):
          list[i][x][y] -= t
        elif (list[i][x][y]<-t):
          list[i][x][y] +=t
        else:
          list[i][x][y] = 0
    return list 

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


infile = sys.argv[1]
outfile = sys.argv[2]
img = cv2.imread(infile)
s = split_rgb(img)
sout = [[[]],[[]],[[]]]
for i in range (0,3):
    nimg = gaussian_noise(s[i])
    g = gauss_pyramid(nimg,5)
    l = lapl_pyramid(g)
    lout = soft_thresholding(l,7)
    sout[i] = collapse(lout)
outimg = img
(outimg[:,:,0], outimg[:,:,1], outimg[:,:,2]) = (sout[0], sout[1], sout[2]) 
print(psnr(s[0],sout[0]))
cv2.imwrite(outfile,outimg)