import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


def gamma_correction(img,gamma):
    b = int(img[0].size/3)
    c = int(img.size/(3*b))
    for x in range(c):
        for y in range(b):
            img[x][y][0] = 255*((img[x][y][0]/255)**(1/gamma))
            img[x][y][1] = 255*((img[x][y][1]/255)**(1/gamma))
            img[x][y][2] = 255*((img[x][y][2]/255)**(1/gamma))
    return img
    
def linear_rescaling(img):
    max0 = 0
    max1 = 0
    max2 = 0
    min0 = 255
    min1 = 255
    min2 = 255
    b = int(img[0].size/3)
    c = int(img.size/(3*b))
    for x in range(c):
        for y in range(b):
            if (img[x][y][0]>max0):
                max0 = img[x][y][0]
            elif(img[x][y][0]<min0):
                min0 = img[x][y][0]
            if (img[x][y][1]>max1):
                max1 = img[x][y][1]
            elif(img[x][y][1]<min1):
                min1 = img[x][y][1]
            if (img[x][y][2]>max2):
                max2 = img[x][y][2]
            elif(img[x][y][2]<min2):
                min2 = img[x][y][2]
    a0 = 255//(max0-min0)
    a1 = 255//(max1-min1)
    a2 = 255//(max2-min2)
    for x in range(c):
        for y in range(b):
            img[x][y][0] = (img[x][y][0]-min0)*a0
            img[x][y][1] = (img[x][y][1]-min1)*a1
            img[x][y][2] = (img[x][y][2]-min2)*a2
    return img

def log_lum (img):
    l1 = 0.114*img[0][0][0]+0.587*img[0][0][1]+0.299*img[0][0][2]
    if (l1>0):
        minlog = math.log10(l1)
        maxlog = math.log10(l1)
    else:
        maxlog = 0
        minlog = 3
    b = int(img[0].size/3)
    c = int(img.size/(3*b)) 
    for x in range(c):
        for y in range(b):
            l = 0.114*img[x][y][0]+0.587*img[x][y][1]+0.299*img[x][y][2]
            if l>0:
                logl = math.log10(l)
                if logl<minlog:
                    minlog = logl
                elif maxlog<logl:
                    maxlog = logl
    print(minlog)
    print(maxlog)
    a = 2.0/(maxlog-minlog)
    for x in range(c):
        for y in range(b):
            l = 0.114*img[x][y][0]+0.587*img[x][y][1]+0.299*img[x][y][2]
            if l>0:
                logl1 = a*(math.log10(l)-minlog)
                l1 = 10**logl1
                img[x][y][0] = img[x][y][0]*l1/l
                img[x][y][1] = img[x][y][1]*l1/l
                img[x][y][2] = img[x][y][2]*l1/l
    return img

def histogram_map(img):
    rows, cols = (1,241) 
    arr = [[0 for i in range(cols)] for j in range(rows)]
    b = int(img[0].size/3)
    c = int(img.size/(3*b))
    for x in range(c):
        for y in range(b):
            l = 0.114*img[x][y][0]+0.587*img[x][y][1]+0.299*img[x][y][2]
            if l>0:
                l1 = math.floor(math.log10(l)*100)
            else:
                l1 = 0
            arr[0][l1] = arr[0][l1]+1
    for x in range(rows):
        for y in range(1,241):
            arr[x][y] = arr[x][y]+arr[x][y-1]
    x = img.size/3
    for a in range(cols):
        arr[0][a] = 240*arr[0][a]/x
    return arr


def histogram_equalisation(img):
    histogram = histogram_map(img)
    b = int(img[0].size/3)
    c = int(img.size/(3*b))
    for x in range(c):
        for y in range(b):
            l = 0.114*img[x][y][0]+0.587*img[x][y][1]+0.299*img[x][y][2]
            if l>0:
                l1 = math.floor(math.log10(l)*100)
            else:
                 l1 = 0
            lf = 10**(histogram[0][l1]/100)
            if l1>0:
                img[x][y][0] = math.floor(img[x][y][0]*lf/l)
                img[x][y][1] = math.floor(img[x][y][1]*lf/l)
                img[x][y][2] = math.floor(img[x][y][2]*lf/l)           
    return img

def gaussian_blur(img):
    b = int(img[0].size)
    c = int(img.size/(b))
    filter = [[1,2,1],[2,4,2],[1,2,1]]
    sum = 0
    imgb = np.arange(img.size).reshape(c,b)
    for x in range(c):
        for y in range (b):
            for row in range(3):
                for col in range(3):
                    if((x-1+row)>=0 and (x-1+row)<c and (y-1+col)>=0 and (y-1+col)<b):
                        sum = sum+img[x-1+row][y-1+col]*filter[row][col]
            imgb[x][y] = int(sum/16)
            sum = 0
    return imgb
    
def grayscale(img):
    b = int(img[0].size/3)
    c = int(img.size/(3*b))
    imgf = np.arange(int(img.size/3)).reshape(c,b)
    for x in range(c):
        for y in range(b):
            imgf[x][y] = 0.114*img[x][y][0]+0.587*img[x][y][1]+0.299*img[x][y][2]
    return (imgf)

def unsharp_masking(img):
    b = int(img[0].size/3)
    c = int(img.size/(3*b))
    gimg = grayscale(img)
    gblur = gaussian_blur(gimg)
    for x in range(c):
        for y in range(b):
            l = 0.114*img[x][y][0]+0.587*img[x][y][1]+0.299*img[x][y][2]
            if l>0 and gblur[x][y]>0:
                l1 = math.log10(l)+(math.log10(l) -math.log10(gblur[x][y]))
            else:
                l1 = 0
            limg = 10**l1 
            if l>0:
                img[x][y][0] = img[x][y][0]*limg/l
                img[x][y][1] = img[x][y][1]*limg/l
                img[x][y][2] = img[x][y][2]*limg/l
    return (img)



infile = sys.argv[1]
outfile = sys.argv[2]
img = cv2.imread(infile)
gamma = 2.2
cv2.imwrite("in.jpg",img)
img = unsharp_masking(img)
cv2.imwrite(outfile,gamma_correction(img,gamma))