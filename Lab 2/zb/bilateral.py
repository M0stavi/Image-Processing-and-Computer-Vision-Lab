# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:23:01 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = "F:/Online Class/4-1/zLabs/Vision/rubiks_cube.png"

img = cv.imread(path)

img = cv.resize(img,(300,300))

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input for bilateral: ")

plt.show()

print("enter kernel height: ")

k_h = int(input())

print("enter kernel weidth: ")

k_w = int(input())

kernel = np.zeros((k_h,k_w), np.float32)

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2

pi = 3.1416

sigma = 75.0

s = 2*sigma*sigma

for i in range(-a,a+1):
    for j in range(-b,b+1):
        r = math.sqrt(i*i+j*j)
        r/=s
        r = math.exp(-(r))
        r/=(pi*s)
        kernel[i+a][j+b] = r
        
img = img/255

m = img.shape[0]
n = img.shape[1]

op = np.zeros((m,n),np.float32)

for i in range(m):
    for j in range(n):
        val = 0.0
        kernel_tem = np.zeros((k_h,k_w), np.float32)
        
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    r = img[i][j] - img[i-x][j-y]
                    r = math.sqrt(r*r)
                    r/=s
                    r=math.exp(-(r))
                    r/=(pi*s)
                    kernel_tem[x+a][y+b]=kernel[x+a][y+b]*r
                   
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    val += kernel_tem[x+a][y+b]*img[i-x][j-y]
        
        ksum = kernel_tem.sum()
        
        val/=ksum
        op[i][j] = val
        
op = op*255

plt.imshow(op,'gray')

plt.title("Output for bilateral: ")

plt.show()