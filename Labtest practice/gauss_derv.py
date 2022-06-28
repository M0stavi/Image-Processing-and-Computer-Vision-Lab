# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 01:08:41 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = 'lena.png'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

m = img.shape[0]
n = img.shape[1]

kx =np.zeros((7,7),np.float32)

ky =np.zeros((7,7),np.float32)

sig = 1.1

s =2*sig*sig
s4 = sig*sig*sig*sig

a = kx.shape[0]//2
b = kx.shape[1]//2

for x in range(-a,a+1):
    for y in range(-b,b+1):
        r = x**2+y**2
        r/=s
        r = np.exp(-(r))
        r*=(-x)
        r /= (2*np.pi*s4)
        kx[a+x][b+y] = r
        
for i in range(7):
    print(kx[i])
    
for x in range(-a,a+1):
    for y in range(-b,b+1):
        r = x**2+y**2
        r/=s
        r = np.exp(-(r))
        r*=(-y)
        r /= (2*np.pi*s4)
        ky[a+x][b+y] = r
        
for i in range(7):
    print(ky[i])
    
op1 = np.zeros((m,n),np.float32)
op2 = np.zeros((m,n),np.float32)

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op1[i][j]+=img[i-x][j-y]*kx[a+x][b+y]
                    
plt.imshow(op1,'gray')
plt.show()

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op2[i][j]+=img[i-x][j-y]*ky[a+x][b+y]
                    
plt.imshow(op2,'gray')
plt.show()
