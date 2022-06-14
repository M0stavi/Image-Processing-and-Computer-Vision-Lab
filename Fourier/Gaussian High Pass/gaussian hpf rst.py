# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:32:32 2022

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

# path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Fourier/Gaussian low pass/input.jpg"

path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Fourier/Homomorphic filter/ln2.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

sigma =10.0



gauss = np.zeros((img.shape[0],img.shape[1]),np.float32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        u = (i-img.shape[0]//2)**2
        v = (j-img.shape[1]//2)**2
        r = math.exp(-((u+v)/(2.0*np.pi*sigma**2)))
        gauss[i][j] = r
        
gauss = 1-gauss

plt.imshow(gauss,'gray')

plt.show()

# fourier transform:
    
f = np.fft.fft2(img)

shift = np.fft.fftshift(f)

mag = np.abs(shift)

angle = np.angle(shift)

mag = mag*gauss

op = np.multiply(mag,np.exp(1j*angle))

# inverse shift and transform:

ishift = np.fft.ifftshift(op)

inv = np.real(np.fft.ifft2(ishift))

plt.imshow(inv,'gray')

plt.show()

inv = clip(inv)

plt.imshow(inv,'gray')

plt.show()