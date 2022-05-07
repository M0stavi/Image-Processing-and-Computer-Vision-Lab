# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:18:51 2022

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

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input")

plt.show()

im_g = img

blur = cv.GaussianBlur(im_g,(3,3),0)

laplacian = cv.Laplacian(blur,cv.CV_64F)

plt.imshow(laplacian,'gray')

plt.title("Function: ")

plt.show()

print("Enter kernel height: ")

k_h = int(input())

print("Enter kernel weidth: ")

k_w = int(input())

kernel1 = np.zeros((k_h,k_w),np.float32)

kernel2 = np.zeros((k_h,k_w),np.float32)

sigma = 2.5

s = sigma*sigma

a = kernel1.shape[0] // 2
b = kernel1.shape[1] // 2

for x in range(-a,a+1):
    for y in range(-b,b+1):
        r = (x*x+y*y)
        r/=(2*s)
        r = math.exp(-(r))
        r = r/(2*3.1416*s)
        kernel1[a+x][b+y] = r

sigma = 1.0

s = sigma*sigma

for x in range(-a,a+1):
    for y in range(-b,b+1):
        r = (x*x+y*y)
        r/=(2*s)
        r = math.exp(-(r))
        r = r/(2*3.1416*s)
        kernel2[a+x][b+y] = r
        
kernel1 = kernel1-kernel2

m = img.shape[0]
n = img.shape[1]

op = np.zeros((m,n),np.float32)

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op[i][j]+=kernel1[a+x][b+y]*img[i-x][j-y]
                else:
                    op[i][j]+=0

plt.imshow(op,'gray')

plt.title("Laplace filtered: ")

plt.show()

op = clip(op)

plt.imshow(op,'gray')

plt.title("Laplace clipped: ")

plt.show()

img = img+op

img=clip(img)

plt.imshow(img,'gray')

plt.title("Laplace enhanced: ")

plt.show()

                    