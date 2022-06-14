# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:47:43 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

def scale(img):
    mn = img.min()
    mx = img.max()
    img = ((img-mn)*(mx-mn))*255
    return img.astype(np.float32)

path = "F:/Online Class/4-1/zLabs/Vision/rubiks_cube.png"

img = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img, 'gray')

plt.title("Input")

plt.show()

print("Enter kernel height: ")

k_h = int(input())

print("Enter kernel weidth: ")

k_w = int(input())

kernel = np.zeros((k_h,k_w), np.float32)

delta = np.zeros((k_h,k_w), np.float32)

sigma = 1

s = 2.0*sigma*sigma

pi = 3.1416

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2

delta[a][b] = 1

for i in range(-a, a+1):
    for j in range(-b,b+1):
        r = (i*i+j*j)
        r/=s
        r = math.exp(-(r))
        r/=(pi*s)
        kernel[i+a][j+b] = r

print("kernel")

for i in range(k_h):
    print(delta[i])
        
m = img.shape[0]
n = img.shape[1]
op = np.zeros((m,n), np.float32)
ksum = kernel.sum()

print('ksum',ksum)

kernel = delta-kernel

for i in range(m):
    for j in range(n):
        val = 0
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    val+= kernel[x+a][y+b]*img[i-x][j-y]
                else:
                    val+=0
        val/=ksum
        op[i][j] = val
        
op = scale(op)

plt.imshow(op,'gray')
plt.title("Output for gaussian blurr")
plt.show()