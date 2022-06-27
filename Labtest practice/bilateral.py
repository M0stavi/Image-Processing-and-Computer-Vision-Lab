# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 19:52:05 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = 'rubiks_cube.png'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

img = cv.resize(img,(250,250))

plt.imshow(img,'gray')

plt.show()

ks = int(input("Enter kernel size: "))

g = np.zeros((ks,ks),np.float32)

a = ks//2
b = ks//2

sig = ks//5

s = 2*sig**2

for i in range(-a,a+1):
    for j in range(-b,b+1):
        r = i*i+j*j
        r/=s
        r = np.exp(-(r))
        r/=(np.pi*s)
        g[i+a][j+b] = r
        
# for i in range(5):
#     print(g[i])
        

m = img.shape[0]
n = img.shape[1]

op = np.zeros((m,n),np.float32)

for i in range(m):
    for j in range(n):
        kernel = np.zeros((ks,ks),np.float32)
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i+x >=0 and i+x<m and j+y>=0 and j+y<n:
                    kernel[a+x][b+y] = g[a+x][b+y]*img[i+x][j+y]
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op[i][j]+=kernel[a+x][b+y]*img[i-x][j-y]


plt.imshow(op,'gray')

plt.show()