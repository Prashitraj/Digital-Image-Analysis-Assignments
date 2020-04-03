import sys
import cv2
import random
import numpy as np
from numpy.fft import fft2, ifft2
import math



def psf(img):
    row,col= img.shape
    output = np.zeros((row,col),np.float32)
    for i in range(20):
            if i<5:
                output[5][i] = 1./5
            if (i>=5 and i<12):
                output[4][i] = 1./7 
            if (i>=12 and i<15):
                output[3][i] = 1./7
            if (i>=15 and i<17):
                output[2][i] = 1./8
            if (i==17 ):
                output[2][i] = 1./8       
    output.astype("uint8")
    return output

def inverse_filtering(degraded_img,kernel):
    fdimg = fft2(degraded_img)
    fk = fft2(kernel)
    hconj = np.conj(fk)
    h2 = np.abs(fk)**2
    fout = fdimg*hconj/h2
    out = ifft2(fout)
    return out

def wiener_filter1(degraded_img, kernel, noise):
    dummy = degraded_img
    dummy = fft2(dummy)
    avg = dummy[0][0]
    sf = np.full(dummy.shape,2*np.sqrt(avg))
    sf = sf**2
    sn = np.abs(fft2(noise))**2
    h = fft2(kernel)
    hconj = np.conj(h)
    h2 = np.abs(h)**2
    g= fft2(degraded_img)
    dummy = (hconj*sf*g)/(sf*h2+sn)
    dummy = np.abs(ifft2(dummy))
    return dummy

def wiener_filter2(degraded_img, kernel, noise):
    dummy = degraded_img
    dummy = fft2(dummy)
    row,col = dummy.shape
    k = 1
    c = dummy[0][0]
    for x in range(row):
        for y in range(col):
            if x!=0 and y!= 0:
                dummy[x][y] = ((x**2+y**2)**(-0.7))*c
    sf = dummy**2
    sn = np.abs(fft2(noise))**2
    h = fft2(kernel)
    hconj = np.conj(h)
    h2 = np.abs(h)**2
    g= fft2(degraded_img)
    dummy = (hconj*sf*g)/(sf*h2+sn)
    dummy = np.abs(ifft2(dummy))
    return dummy

infile = sys.argv[1]
outfile = sys.argv[2]
dimg = cv2.imread(infile,cv2.IMREAD_GRAYSCALE)
print(dimg.shape)
cv2.imshow('degraded_image',dimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
np.array(dimg).astype("uint8")
kernel = psf(dimg)
noise  = np.random.normal(0,4,np.shape(dimg))
outimg2 = wiener_filter1(dimg,kernel,noise)
cv2.imwrite(outfile,outimg2)
fimg = cv2.imread(outfile,cv2.IMREAD_GRAYSCALE)
cv2.imshow('FILTERED_image',fimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
