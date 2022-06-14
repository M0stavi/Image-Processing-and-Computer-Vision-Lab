# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 19:02:38 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = 'C:/Users/Asus/Desktop/erd.jpg'

img = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input for erotion")

plt.show()

t, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)

img=img//255

plt.imshow(img,'gray')

plt.title("Thresholded Image")

plt.show()

# erotion

kernel = np.ones((3,3),np.float32)

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2

m = img.shape[0]
n = img.shape[1]

op = np.zeros((m,n), np.float32)

for i in range(m):
    for j in range(n):
        flag = 1
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if x+i>=0 and x+i<m and j+y>=0 and j+y<n:
                    if img[i+x][j+y] != kernel[a+x][b+y]:
                        flag = 0
        if flag:
            op[i][j] = 1
        else:
            op[i][j] = 0


plt.imshow(op,'gray')

plt.title("Erotion output")

plt.show()