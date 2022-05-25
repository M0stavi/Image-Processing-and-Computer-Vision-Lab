# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:40:34 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def clip(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 0:
                img[i][j] = 0
            if img[i][j] > 255:
                img[i][j] = 255
    return img.astype(np.float32)

path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Fourier/Homomorphic filter/ln2.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

homo = np.zeros((img.shape[0],img.shape[1]), np.float32)

gh = 1.2
gl = 0.5
c = 0.1
d0 = 50

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        u = (i-img.shape[0]/2)**2
        v = (j-img.shape[1]/2)**2
        r = math.exp(-c*((u+v)/d0**2))
        r = (gh-gl)*(1-r)+gl
        homo[i][j] = r
        

plt.imshow(homo,'gray')

plt.show()

img = np.log1p(img)

# fourier transform:
    
f = np.fft.fft2(img)

shift = np.fft.fftshift(f)

mag = np.abs(shift)

angle = np.angle(shift)

mag = mag*homo

op = np.multiply(mag,np.exp(1j*angle))

# inverse shift and transform:

ishift = np.fft.ifftshift(op)

inv = np.real(np.fft.ifft2(ishift))

inv = np.exp(inv)-1

plt.imshow(inv,'gray')

plt.show()

inv = clip(inv)

plt.imshow(inv,'gray')

plt.show()