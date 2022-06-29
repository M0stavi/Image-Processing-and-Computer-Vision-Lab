# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 04:10:06 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# import pypher

path = 'md.png'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

m = img.shape[0]
n = img.shape[1]

k = np.zeros((5,5),np.float32)

for i in range(5):
    k[i][i] = 1
    
a = k.shape[0]//2
b = k.shape[1]//2
    
# bl = cv.filter2D(img, ddepth=-1,kernel=k)

bl = np.zeros((m,n),np.float32)

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    bl[i][j]+=img[i-x][j-y]*k[a+x][b+y]
    
    
op = np.zeros((m,n),np.float32)

x = img.shape[0]//2-k.shape[0]//2
kp = np.pad(k,(x,x),'constant',constant_values=0)

bl = cv.resize(bl,(kp.shape[0],kp.shape[1]))

print(kp.shape)

# bl = cv.filter2D(img, ddepth=-1,kernel=kp)

plt.imshow(bl,'gray')

plt.show()

g = np.fft.fft2(bl)
h = np.fft.fft2(kp)



F_hat = np.divide(g,h)

f_hat = np.real(np.fft.ifft2(F_hat))

op = np.fft.fftshift(f_hat)

plt.imshow(op,'gray')

plt.show()

# for i in range()