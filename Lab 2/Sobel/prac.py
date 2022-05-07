# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:40:03 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

def clip(img):
    m = img.shape[0]
    n = img.shape[1]
    for i in range(m):
        for j in range(n):
            if img[i][j] > 255:
                img[i][j] = 255
            if img[i][j] < 0:
                img[i][j] = 0
    return img.astype(np.float32)

path = "C:/Users/Asus/Desktop/1.png"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input for sobel: ")

plt.show()

kernel = np.array(([-1,0,1],[-2,0,2],[-1,0,1]),np.float32)

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2

m = img.shape[0]
n = img.shape[1]

op1 = np.zeros((m,n),np.float32)

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op1[i][j]+=kernel[a+x][b+y]*img[i-x][j-y]
                else:
                    op1[i][j]+=0

plt.imshow(op1,'gray')

plt.title("Sobel vertical filtered: ")

plt.show()

op1 = clip(op1)

plt.imshow(op1,'gray')

plt.title("Sobel vertical clipped: ")

plt.show()

op2 = np.zeros((m,n),np.float32)

kernel = np.array(([-1,-2,-1],[0,0,0],[1,2,1]),np.float32)

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op2[i][j]+=kernel[a+x][b+y]*img[i-x][j-y]
                else:
                    op2[i][j]+=0

plt.imshow(op2,'gray')

plt.title("Sobel horizontal filtered: ")

plt.show()

op1 = clip(op2)

plt.imshow(op2,'gray')

plt.title("Sobel horizontal clipped: ")

plt.show()

op2+=op1

op2 = clip(op2)

plt.imshow(op2,'gray')

plt.title("Sobel added: ")

plt.show()

im_o = img+op2

im_o = clip(im_o)

plt.imshow(im_o,'gray')

plt.title("Sobel enhanced:")

plt.show()