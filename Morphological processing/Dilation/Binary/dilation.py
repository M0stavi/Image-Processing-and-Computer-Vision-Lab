# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 20:47:58 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = 'C:/Users/Asus/Desktop/erd.jpg'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input for dilation: ")

plt.show()

t, img = cv.threshold(img,100,255,cv.THRESH_BINARY)

img = img//255

plt.imshow(img,'gray')

plt.title("Thresholded image:")

plt.show()

m = img.shape[0]
n = img.shape[1]

k_h = int(input("Enter kernel height: "))

k_w = int(input("Enter kernel weidth: "))

a = k_h // 2
b = k_w // 2

kernel = np.ones((k_h,k_w),np.float32)
op = np.zeros((m,n),np.float32)

# dilation

for i in range(m):
    for j in range(n):
        flag = 0
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    if img[i-x][j-y] == kernel[a+x][b+y]:
                        flag = 1
        if flag:
            op[i][j] = 1
        else:
            op[i][j] = 0
        
            
plt.imshow(op,'gray')

plt.title("Output for dilation")

plt.show()