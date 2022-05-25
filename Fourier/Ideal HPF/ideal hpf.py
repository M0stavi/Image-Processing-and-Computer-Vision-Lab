# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:37:35 2022

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

path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Fourier/Gaussian low pass/input.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

lpf = np.zeros((img.shape[0],img.shape[1]),np.float32)

d0 = 20.0

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        u = (i-img.shape[0]//2)**2
        v = (j-img.shape[1]//2)**2
        d = math.sqrt(u+v)
        # print("D: ", d)
        if d<=d0:
            lpf[i][j] = 1.0
            
lpf = 1-lpf
            

plt.imshow(lpf,'gray')

plt.show()

# fourier transform:
    
f = np.fft.fft2(img)

shift = np.fft.fftshift(f)

mag = np.abs(shift)

angle = np.angle(shift)

mag = mag*lpf

op = np.multiply(mag,np.exp(1j*angle))

inv = np.fft.ifftshift(op)

inv = np.real(np.fft.ifft2(inv))

plt.imshow(inv,'gray')

plt.show()

inv = clip(inv)

plt.imshow(inv,'gray')

plt.show()