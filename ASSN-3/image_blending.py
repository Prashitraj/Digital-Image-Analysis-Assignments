import sys
import os
import numpy as np
import cv2
import scipy
from scipy.stats import norm
from scipy.signal import convolve2d
import math


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
'''Blend the two laplacian pyramids by weighting them according to the mask.'''
def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
  blended_pyr = []
  k= len(gauss_pyr_mask)
  for i in range(0,k):
   p1= gauss_pyr_mask[i]*lapl_pyr_white[i]
   p2=(1 - gauss_pyr_mask[i])*lapl_pyr_black[i]
   blended_pyr.append(p1 + p2)
  return blended_pyr
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

infile1 = sys.argv[1]
infile2 = sys.argv[2]
rfile = sys.argv[3]
outfile = sys.argv[4]
img1 = cv2.imread(infile1)
img2 = cv2.imread(infile2)
region = cv2.imread(rfile,cv2.IMREAD_GRAYSCALE)
region= region/255
s1 = split_rgb(img1)
s2 = split_rgb(img2)
sout = [[[]],[[]],[[]]]
for i in range (0,3):
  g1 = gauss_pyramid(s1[i],5)
  g2 = gauss_pyramid(s2[i],5)
  gr = gauss_pyramid(region,5)
  l1 = lapl_pyramid(g1)
  l2 = lapl_pyramid(g2)
  lout = blend(l1,l2,gr)
  sout[i] = collapse(lout)
outimg = img1
(outimg[:,:,0], outimg[:,:,1], outimg[:,:,2]) = (sout[0], sout[1], sout[2]) 
cv2.imwrite(outfile,outimg)