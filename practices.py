# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:46:06 2022

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
            if img[i][j]>255:
                img[i][j]=255
            if img[i][j] < 0:
                img[i][j] = 0
    return img.astype(np.float32)

def scale(img):
    g_m = img-img.min()
    g_s = 255*(g_m/g_m.max())
    return g_s.astype(np.float32)

path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/test.jpg"

img = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img, 'gray')

plt.title("Input:")

plt.show()

kernel = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]), np.float32)

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

op = clip(op)

plt.imshow(op,'gray')

plt.title("Laplace filtered:")

plt.show()

img = img+op #center of kernel is positive

img = clip(img)

plt.imshow(img,'gray')

plt.title("Laplace enhanced:")

plt.show()

img = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kernel = np.array(([-1,-2,-1],[0,0,0],[1,2,1]), np.float32)

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

op = clip(op)

plt.imshow(op,'gray')

plt.title("Sobel Horizontal:")

plt.show()

op2 = np.zeros((m,n), np.float32)

kernel = np.array(([-1,0,1],[-2,0,2],[-1,0,1]), np.float32)

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op2[i][j]+=kernel[a+x][b+y]*img[i-x][j-y]
                else:
                    op2[i][j]+=0
op2 = clip(op2)

plt.imshow(op2,'gray')

plt.title("Sobel vertical:")

plt.show()

op+=op2

op=clip(op)

plt.imshow(op,'gray')

plt.title("Sobel plus:")

plt.show()

img=img+op

img=clip(img)

plt.imshow(img,'gray')

plt.title("Sobel enhanced:")

plt.show()

                    