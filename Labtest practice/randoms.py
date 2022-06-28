# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:06:05 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def scale(img):
    g_m = img-img.min()
    g_s = (g_m/g_m.max())*255
    return g_s.astype(np.float32)

path ='md.png'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()


k_h = 7

kernel = np.zeros((k_h,k_h),np.float32)

for i in range(k_h):
    kernel[i][i] = 1
    
plt.imshow(kernel,'gray')

plt.show()
  
m = img.shape[0]
n = img.shape[0]

a = kernel.shape[0]//2
b = kernel.shape[0]//2
    
op = np.zeros((m,n),np.float32)

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op[i][j]+=img[i-x][j-y]*kernel[a+x][b+y]
                else:
                    op[i][j]+=0
                    
plt.imshow(op,'gray')

plt.show()

x = int(img.shape[0]//2-kernel.shape[0]//2)
y = int(img.shape[1]//2-kernel.shape[1]//2)

print(x,y)

    
h = np.pad(kernel,(x,y),'constant',constant_values=0)


# h = cv.rotate(h, cv.ROTATE_90_CLOCKWISE)


img = cv.resize(img,(358,358))

print(img.shape)

print(h.shape)

h = np.fft.fft2(h)

for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        if h[i][j] < 20:
            h[i][j] = .000000001

im = np.fft.fft2(img)

# h = np.fft.fftshift(h)

mh = np.abs(h)

mi = np.fft.fftshift(im)

mi =np.abs(mi)

phase = np.angle(mi)

plt.imshow(np.log(mi),'gray')

plt.show()

mi = mi/mh

op = np.multiply(mi,np.exp(1j*phase))

op = np.fft.ifftshift(op)

op = np.real(np.fft.ifft2(op))

op = scale(op)

plt.imshow(op,'gray')

plt.show()