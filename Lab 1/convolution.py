# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 12:43:42 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

path = "F:/Online Class/4-1/zLabs/Vision/lab1/lena.png"

image = cv.imread(path, cv.IMREAD_GRAYSCALE)

plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

plt.show()

kernel = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]), dtype="float32")

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2
m = image.shape[0] 
n = image.shape[1]

op = np.zeros((m,n), np.float32)

for i in range(0,m):
    for j in range(0,n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x >=0 and i-x < m and j-y>=0and j-y < n:
                    op[i][j] += kernel[a+x][b+y]*image[i-x][j-y]
                else:
                    op[i][j] += 0
plt.imshow(cv.cvtColor(op, cv.COLOR_BGR2RGB))

plt.show()                