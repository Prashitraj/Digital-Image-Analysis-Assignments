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

def gaussian_fn(ai,s):
    arr = [[0,0,0],[0,0,0],[0,0,0]]
    for x in range(s):
        for y in range(s):
            dist = (x-1)*(x-1)+(y-1)*(y-1)
            ais = ai*s*ai*s
            arr[x][y] = math.exp(-(dist)/ais)/(math.pi*ais)
    return arr


def grayscale(img):
    b = int(img[0].size/3)
    c = int(img.size/(3*b))
    imgf = np.arange(int(img.size/3)).reshape(c,b)
    for x in range(c):
        for y in range(b):
            imgf[x][y] = 0.06*img[x][y][0]+0.67*img[x][y][1]+0.27*img[x][y][2]
    return (imgf)

def world_lum(wimg,a):
    b = int(wimg[0].size)
    c = int(wimg.size/(b))
    sum = 0
    nimg = [[0 for i in range(b)] for j in range(c)]
    for x in range(c):
        for y in range(b):
            if wimg[x][y]>0:
                nimg[x][y] = math.log10(wimg[x][y])
            else:
                nimg[x][y] = 0
            sum  = sum+nimg[x][y]
    avg = sum/(b*c)
    for x in range(c):
        for y in range(b):
            nimg[x][y] = a*nimg[x][y]/avg
    return (nimg)

def initial_mapping(gimg):
    b = int(len(gimg[0]))
    c = int(len(gimg))
    nimg = [[0 for i in range(b)] for j in range(c)]
    for x in range(c):
        for y in range(b):
            nimg[x][y] = gimg[x][y]/(1+gimg[x][y])
    return (nimg)


def centre(s,gimg,row,col,ai):
    sum= 0
    sumw = 0
    b = int(len(gimg[0]))
    c = int(len(gimg))
    arr = gaussian_fn(ai,s)
    for x in range(s):
        for y in range(s):
            k = (s-1)/2
            if((x-k+row)>=0 and (x+k+row)<c and (y-k+col)>=0 and (y+k+col)<b):
                k = int(k)
                sum = gimg[row-x-k][col-y-k]*arr[x][y]
            sumw = sumw+arr[x][y]
    return(sum/sumw)

def centre_surround(s,gimg,row,column,a,phi):
    v1 = centre(s,gimg,row,column,0.35)
    v2 = centre(s,gimg,row,column,0.56)
    v = (v1-v2)/((2**phi)*a*s*s+v1)
    return v

def min_s(nimg,row,column,e,a):
    s= 1
    fact = 1.6
    sharp = 10
    min = centre_surround(1,nimg,row,column,a,sharp)
    sf = 1
    while(s<20):
        if (min<centre_surround(s,nimg,row,column,a,sharp)):
            min = centre_surround(s,nimg,row,column,a,sharp)
            sf = s
        if (abs(min)<e):
            break
        s = int(s*1.6)
    v1 = centre(sf,nimg,row,column,0.35)
    return v1

def final_lum(nimg,e,a):
    b = int(len(nimg[0]))
    c = int(len(nimg))
    fimg = [[0 for i in range(b)] for j in range(c)]
    for x in range(c):
        for y in range(b):
            fimg[x][y] = 255*nimg[x][y]/(1+min_s(nimg,x,y,e,a))
    return fimg

def log_lum (img,f):
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
                l1 = f*(10**logl1)
                img[x][y][0] = img[x][y][0]*l1/l
                img[x][y][1] = img[x][y][1]*l1/l
                img[x][y][2] = img[x][y][2]*l1/l
    return img

def tone_mapping(img,a):
    b = int(img[0].size/3)
    c = int(img.size/(b*3))
    gimg = grayscale(img)
    wimg = world_lum(gimg,a)
    mimg = initial_mapping(wimg)
    fimg = final_lum(mimg,0.02,a)
    for x in range(c):
        for y in range(b):
            l = 0.06*img[x][y][0]+0.67*img[x][y][1]+0.27*img[x][y][2]
            lf = fimg[x][y]
            if l>0:
                img[x][y][0] = img[x][y][0]*lf/l
                img[x][y][1] = img[x][y][1]*lf/l
                img[x][y][2] = img[x][y][2]*lf/l
    imgn = log_lum(img,2)
    gamma= 1
    imgg = gamma_correction(imgn,gamma)
    return imgg

infile = sys.argv[1]
outfile = sys.argv[2]
img = cv2.imread(infile)
cv2.imwrite("in.jpg",img)
gamma = 2.2
a = 0.18
img = tone_mapping(img,a)
cv2.imwrite(outfile,img)