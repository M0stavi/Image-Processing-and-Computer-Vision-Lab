# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 23:08:45 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn, fftshift

path = 'C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Lab 4/assignment/assm.png'

fxy = cv.imread(path)

fxy = cv.cvtColor(fxy, cv.COLOR_BGR2GRAY)

plt.imshow(fxy,'gray')

plt.title("Input for motion blur")

plt.show()

gauss = np.zeros((5,5),np.float32)

sigma = 1

s = 2*sigma*sigma

a = gauss.shape[0]//2

b = gauss.shape[1]//2

for i in range(-a,a+1):
    for j in range(-b,b+1):
        r = i*i+j*j
        r = r/s
        r = np.exp(-(r))
        r/=(np.pi*s)
        
        gauss[a+i][b+j] = r
        
# print(gauss)

m = fxy.shape[0]
n = fxy.shape[1]

a = int(fxy.shape[0]//2 - gauss.shape[0]//2)

gauss_pad = np.pad(gauss, (a,a-1), 'constant', constant_values=(0))

print(gauss_pad.shape[0], gauss_pad.shape[1])

fxy = cv.resize(fxy,(gauss_pad.shape[0],gauss_pad.shape[0]))

# fourier transform

f = np.fft.fft2(fxy)

h = np.fft.fft2(gauss_pad)

fs = np.fft.fftshift(f)

hs = np.fft.fftshift(h)

fs = np.