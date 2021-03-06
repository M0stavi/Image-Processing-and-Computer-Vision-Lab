# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:06:05 2022

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

gauss = np.random.normal(0,20,img.size)

gauss = gauss.reshape(img.shape[0],img.shape[1])
# x = cv.add(img,gauss)
img = img+gauss

plt.imshow(img,'gray')

plt.show()

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

nsr = 0.0000000005

mg = np.abs(h)

F_hat = np.divide(g,h)

F_hat /=(1+(nsr/(mg**2)))

f_hat = np.real(np.fft.ifft2(F_hat))

op = np.fft.fftshift(f_hat)

plt.imshow(op,'gray')

plt.show()

# for i in range()
                    
        