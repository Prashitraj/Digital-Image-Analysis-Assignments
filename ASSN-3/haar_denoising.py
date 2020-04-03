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
  return red, green, blue
 
'''generate a 5x5 kernel'''
def generating_kernel(a):
  w_1d = np.array([0.30 - a/2.0, 0.20, a, 0.20, 0.30 - a/2.0])
  return np.outer(w_1d, w_1d)

def ireduce(image):
  out = None
  kernel = generating_kernel(0.5)
  outimage = scipy.signal.convolve2d(image,kernel,'same')
  out = outimage[::2,::2]
  return out

def haar_transform(img,m):
    row,col = img.shape
    out1 = np.arange(row*col/4).reshape(row/2,col/2)
    out2 = np.arange(row*col/4).reshape(row/2,col/2)
    out3 = np.arange(row*col/4).reshape(row/2,col/2)
    out4 = np.arange(row*col/4).reshape(row/2,col/2)
    for x in range(row/2):
      for y in range(col/2):
        out1[x][y] = m[0][0]*img[2*x][2*y]+m[0][1]*img[2*x][2*y+1]+m[0][2]*img[2*x+1][2*y]+m[0][3]*img[2*x+1][2*y+1]
        out2[x][y] = m[1][0]*img[2*x][2*y]+m[1][1]*img[2*x][2*y+1]+m[1][2]*img[2*x+1][2*y]+m[1][3]*img[2*x+1][2*y+1]
        out3[x][y] = m[2][0]*img[2*x][2*y]+m[2][1]*img[2*x][2*y+1]+m[2][2]*img[2*x+1][2*y]+m[2][3]*img[2*x+1][2*y+1]
        out4[x][y] = m[3][0]*img[2*x][2*y]+m[3][1]*img[2*x][2*y+1]+m[3][2]*img[2*x+1][2*y]+m[3][3]*img[2*x+1][2*y+1]
    return (out1,out2,out3,out4)


'''create a gaussain pyramid of a given image'''
def haar_pyramid(image, levels,a):
  output = []
  tmp1 = image
  for i in range(0,levels):
    tmp1,tmp2,tmp3,tmp4 = haar_transform(tmp1,a)
    output.append(tmp4)
    output.append(tmp2)
    output.append(tmp3)
  output.append(tmp1)
  return output

def inv_haar_transform(in1,in2,in3,in4,finv):
  row,col = in1.shape

  out = np.arange(4*row*col).reshape(2*row,2*col)
  for x in range(0,row):
    for y in range (0,col):
      out[2*x][2*y] = finv[0][0]*in1[x][y]+finv[0][1]*in2[x][y]+finv[0][2]*in3[x][y]+finv[0][3]*in4[x][y]
      out[2*x+1][2*y] = finv[1][0]*in1[x][y]+finv[1][1]*in2[x][y]+finv[1][2]*in3[x][y]+finv[1][3]*in4[x][y]
      out[2*x][2*y+1] = finv[2][0]*in1[x][y]+finv[2][1]*in2[x][y]+finv[2][2]*in3[x][y]+finv[2][3]*in4[x][y]
      out[2*x+1][2*y+1] = finv[3][0]*in1[x][y]+finv[3][1]*in2[x][y]+finv[3][2]*in3[x][y]+finv[3][3]*in4[x][y] 
  return out
  

def retain_image(out,finv,levels):
  l = len(out)
  out1 = out[l-1]
  for i in range (0,levels):
    out1 = inv_haar_transform(out1,out[l-2-3*i],out[l-3-3*i],out[l-4-3*i],finv)
  return out1

def hard_thresholding (list,t):
  for i in range (len(list)-1):
    row,col = list[i].shape
    for x in range(row):
      for y in range(col):
        if (abs(list[i][x][y])<t):
          list[i][x][y] = 0
  return list

def soft_thresholding (list,t):
  for i in range (len(list)-1):
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
a = np.array([[0.25, 0.25, 0.25, 0.25], [1.,-1.,1.,-1.],[1.,1.,-1.,-1.],[1.,-1.,-1.,1.]])
ainv = np.linalg.inv(a)
s = split_rgb(img)
sout = [[[]],[[]],[[]]]
for i in range (0,3):
    nimg = gaussian_noise(s[i])
    h = haar_pyramid(nimg,5,a)
    hout = soft_thresholding(h,20)
    sout[i] = retain_image(hout,ainv,5)
outimg = img
(outimg[:,:,0], outimg[:,:,1], outimg[:,:,2]) = (sout[0], sout[1], sout[2]) 
print(psnr(s[0],sout[0]))
cv2.imwrite(outfile,outimg)