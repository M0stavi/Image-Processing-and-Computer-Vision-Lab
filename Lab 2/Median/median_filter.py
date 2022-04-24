# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 20:27:41 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = "F:/Online Class/4-1/zLabs/Vision/ImageProcessing-fromScratchWithOpenCV-/SpatialFiltering/Median/input.jpg"

img  = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img, "gray")

plt.title("Input for median filter:")

plt.show()

print("Enter kernel height: ")

k_h = int(input())

print("Enter kernel width: ")

k_w = int(input())

kernel = np.ones((k_h,k_w), np.float32) #box kernel

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2
m = img.shape[0]
n = img.shape[1]
total = k_h*k_w

op = np.zeros((m,n), np.float32)

for i in range(m):
    for j in range(n):
        values = []
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i+x>=0 and i+x<m and j+y>=0 and j+y<n:
                    values.append(kernel[a+x][b+y]*img[i+x][j+y])
                else:
                    values.append(0)
        values.sort()
        median = len(values) // 2
        op[i][j] = values[median]
        op[i][j]/=total
        
plt.imshow(op, "gray")

plt.title("Output for median: ")

plt.show()


                
        

