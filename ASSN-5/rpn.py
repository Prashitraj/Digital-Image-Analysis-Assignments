import cv2 
import numpy as np
import math

def p2r(radii, angles):
    return radii * np.exp(1j*angles)

def r2p(x):
    return np.abs(x), np.angle(x)


def i2p(img):
    fimg=np.fft.fft2(img)
    row = len(img)
    col = len(img[0])

    pimg=np.zeros([row,col],dtype=np.complex)
    for i in range(row):
        for j in range(col):
            if (i==0 and j==0):
                pimg[i][j]=np.sum(np.ravel(img))
            else:
                pimg[i][j]=fimg[i][j]/(2*np.cos(2*i*math.pi/row)+2*np.cos(2*j*math.pi/col)-4)
    pout=np.fft.ifft2(pimg)
    return(pout)

def ExpandTexture(img):
    row=len(img)
    col=len(img[0])
    m=np.mean(img.ravel())
    pimg = i2p(img)
    P=np.abs(pimg)
    TextureFFT=np.fft.fft2(img)
    norm, phase = r2p(TextureFFT)
    RandomPhase=np.random.rand(row,col)*math.pi-np.ones([row,col])*math.pi
    NTextureFFT=p2r(norm,phase+RandomPhase)
    return(np.abs(np.fft.ifft2(NTextureFFT)))

img = cv2.imread('15.jpg',cv2.IMREAD_GRAYSCALE)
out = ExpandTexture(np.array(img,dtype=np.float))
cv2.imwrite('out.png',out)