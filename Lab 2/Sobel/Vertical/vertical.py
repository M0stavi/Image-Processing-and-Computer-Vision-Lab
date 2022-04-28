# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 23:25:28 2022

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

plt.title("Input for sobel vertical: ")

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

plt.imshow(op, 'gray')
plt.title("Output for sobel vertical: ")
plt.show()