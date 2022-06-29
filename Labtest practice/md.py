# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 04:10:06 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = 'md.png'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

m = img.shape[0]
n = img.shape[1]

ks= 5

k = np.zeros((ks,ks),np.float32)

for i in range(ks):
    k[i][i] = 1
    
op = np.zeros((m,n),np.float32)

a = ks//2
b = ks//2

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op[i][j]+=img[i-x][j-y]*k[a+x][b+y]
                    
                    
plt.imshow(op,'gray')

plt.show()

pp = m//2-a

kp = np.pad(k,(pp,pp),'constant',constant_values=0)

img = cv.resize(img,(kp.shape[0],kp.shape[1]))

print(kp.shape,img.shape)

f = np.fft.fft2(img)

h = np.fft.fft2(kp)

for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        if h[i][j]<20:
            h[i][j] = .0000001

op = np.divide(f,h)

op = np.real(np.fft.ifft2(op))
# op = np.fft.ifftshift(op)

plt.imshow(op,'gray')

plt.show()