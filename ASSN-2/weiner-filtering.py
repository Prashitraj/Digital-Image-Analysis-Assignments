import sys
import cv2
import random
import numpy as np
import math

def gaussian_noise(img):
    sigma = 10
    gauss = np.random.normal(0, sigma, np.shape(img))
    noisy_img = img + gauss
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img

def sp_noise(image,prb):
    output = np.zeros(image.shape,np.uint8)
    t = 1 - prb 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prb:
                output[i][j] = 0
            elif rdn > t:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def mean_filter(img,fsize):
    kernel = np.ones((fsize,fsize),np.float32)/(fsize**2)
    dst = cv2.filter2D(img,-1,kernel)
    return dst

def theshold(image):
    output = np.zeros(image.shape,np.uint8)
    row,col = image.shape
    for x in range (row):
        for y in range(col):
            if(image[x][y]<0):
                output[x][y] = 0
            elif(image[x][y]>255):
                output[x][y] = 255
            else:
                output[x][y] = math.floor(image[x][y]) 
    return output

def median_filter(img,fsize):
    median = cv2.medianBlur(img, fsize)
    return median

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def best_mean_filter(img):
    n = 0
    imgf = img
    psn = 0
    fsize = 0
    for i in range(4):
        img1 = mean_filter(img,2*i+3)
        psn1 = psnr(img1,img)
        print(psn1)
        if (psn1>psn):
            imgf = img1
            fsize = 2*i+3
            psn = psn1
    print(psn,fsize)
    return imgf

def best_median_filter(img):
    n = 0
    imgf = img
    psn = 0
    fsize = 0
    for i in range(4):
        img1 = median_filter(img,2*i+3)
        psn1 = psnr(img1,img)
        print(psn1)
        if (psn1>psn):
            imgf = img1
            fsize = 2*i+3
            psn = psn1   
    print (psn,fsize)
    return imgf


infile = sys.argv[1]
midfile = sys.argv[2]
outfile = sys.argv[3]
img = cv2.imread(infile,cv2.IMREAD_GRAYSCALE)
noisy = sp_noise(img,0.05)
noisy = theshold(noisy)
cv2.imwrite(midfile,noisy)
outimg = best_median_filter(noisy) 
cv2.imwrite(outfile,outimg)