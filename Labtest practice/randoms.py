# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:06:05 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = 'homo.jpg'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

mm = img.shape[0]

nn = img.shape[1]

h = np.zeros((mm,nn),np.float32)

d0=50
c=0.1
gh=1.2
gl=0.5

for i in range(mm):
    for j in range(nn):
        x = (i-mm//2)**2
        y = (j-nn//2)**2
        d = x+y
        r = (gh-gl)*(1-np.exp(-(c*(d/d0**2))))+gl
        h[i][j]= r
        
plt.imshow(h,'gray')

plt.show()

f = np.fft.fft2(img)

sft=np.fft.fftshift(f)

mag = np.abs(sft)

phase = np.angle(sft)

mag = mag*h

op=np.multiply(mag,np.exp(1j*phase))

op= np.fft.ifftshift(op)

op = np.real(np.fft.ifft2(op))

plt.imshow(op,'gray')

plt.show()