import sys
import cv2
import random
import numpy as np
from numpy.fft import fft2, ifft2
import math


def add_gaussian_noise(img, sigma):
	gauss = np.random.normal(0, sigma, np.shape(img))
	noisy_img = img + gauss
	noisy_img[noisy_img < 0] = 0
	noisy_img[noisy_img > 255] = 255
	return noisy_img

def magnitude(noise):
    sum1 = 0
    row1,col1 = noise.shape
    for x in range(row1):
        for y in range(col1):
            sum1 = sum1+(np.abs(noise[x][y]))**2
    return sum1

def disc_psf(img):
    row,col= img.shape
    output = np.zeros((row,col),np.float32)
    k = 0
    for i in range(10):
        for j in range(10):
            if ((i-5)**2+(j-5)**2)<=20:
                output[i][j] = 1.
                k=k+1
    output = output/k
    print(k)
    return output

def blur(img, kernel_size = 3):
	dummy = np.copy(img)
	h = np.eye(kernel_size) / kernel_size
	dummy = cv2.filter2D(dummy,-1, h)
	return dummy

# weiner filter with F(u,v) is of original spectrum
def wiener_filter(img,degraded_img, kernel, noise):
    sf = (np.abs(fft2(img)))**2
    sn = (np.abs(fft2(noise)))**2
    h = fft2(kernel)
    hconj = np.conj(h)
    h2 = np.abs(h)**2
    g= fft2(degraded_img)
    dummy = (hconj*sf*g)/(sf*h2+sn)
    dummy = np.abs(ifft2(dummy))
    return dummy

# weiner filter with F(u,v) is constant
def wiener_filter1(degraded_img, kernel, noise):
    dummy = degraded_img
    dummy = fft2(dummy)
    avg = dummy[0][0]
    dummy = np.full(dummy.shape,2*np.sqrt(avg))
    sf = dummy**2
    sn = np.abs(fft2(noise))**2
    h = fft2(kernel)
    hconj = np.conj(h)
    h2 = np.abs(h)**2
    g= fft2(degraded_img)
    dummy = (hconj*sf*g)/(sf*h2+sn)
    dummy = np.abs(ifft2(dummy))
    return dummy

# weiner filter with F(u,v) prpotional to (u**2+v**2)**alpha
def wiener_filter2(degraded_img, kernel, noise):
    dummy = degraded_img
    dummy = fft2(dummy)
    row,col = dummy.shape
    k = 1
    c = dummy[0][0]
    for x in range(row):
        for y in range(col):
            if x!=0 and y!= 0:
                dummy[x][y] = ((x**2+y**2)**(-0.65))*c
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
outfile1 = sys.argv[2]
outfile = sys.argv[3]
img = cv2.imread(infile,cv2.IMREAD_GRAYSCALE)
kernel = disc_psf(img)
outimg1 = ifft2(fft2(img)*fft2(kernel))
degraded_img = np.abs(add_gaussian_noise(outimg1,5))
np.array(degraded_img).astype("uint8")
cv2.imwrite(outfile1,degraded_img)
noise  = np.random.normal(0,5,np.shape(img))
outimg2 = wiener_filter(img,degraded_img,kernel,noise)
cv2.imwrite(outfile,outimg2)

for x in range(5):
    noise  = np.random.normal(0,(5+x),np.shape(img))
    mag1 =magnitude(noise)
    mag2 =  magnitude(fft2(noise)) 
    print(mag1, mag2,mag1/mag2)