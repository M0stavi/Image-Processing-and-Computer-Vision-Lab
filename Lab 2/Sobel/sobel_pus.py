# -*- coding: utf-8 -*-
"""
Created on Sun May  1 16:13:05 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = "F:/Online Class/4-1/zLabs/Vision/ein.jpg"

img = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input for sobel combined: ")

plt.show()

kernel = np.array(([-1,0,1],[-2,0,2],[-1,0,1]), np.float32)

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2
m = img.shape[0]
n = img.shape[1]

op = np.zeros((m,n), np.float32)

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op[i][j]+=kernel[a+x][b+y]*img[i-x][j-y]
                else:
                    op[i][j]+=0

kernel2 = np.array(([-1,-2,-1], [0,0,0],[1,2,1]), np.float32)

a = kernel2.shape[0] // 2
b = kernel2.shape[1] // 2
m = img.shape[0]
n = img.shape[1]

op2 = np.zeros((m,n), np.float32)

for i in range(m):
    for j in range(n):
        for x in range (-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op2[i][j] += kernel2[x+a][y+b]*img[i-x][j-y]
                else:
                    op2[i][j]+=0

op = op+op2

for i in range(m):
    for j in range(n):
        if op[i][j] > 255:
            op[i][j] = 255
        if op[i][j] < 0:
            op[i][j] = 0
            
plt.imshow(op, 'gray')
plt.title("Output for sobel combined: ")
plt.show()

img = img+op

for i in range(m):
    for j in range(n):
        if img[i][j] > 255:
            img[i][j] = 255
        if img[i][j] < 0:
            img[i][j] = 0
            
plt.imshow(img, 'gray')
plt.title("Enhanced with sobel: ")
plt.show()


