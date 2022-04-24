# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 21:42:13 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Lab 2/slide.png"

img = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img, "gray")

plt.title("Input: ")

plt.show()

print("Enter kernel height: ")

k_h = int(input())

print("Enter kernel weidth: ")

k_w = int(input())

kernel = np.ones((k_h,k_w), np.float32)

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2
m = img.shape[0]
n = img.shape[1]

op = np.zeros((m,n), np.float32)

ksum = kernel.sum()

for i in range(m):
    for j in range(n):
        value = 0
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i+x >= 0 and i+x <m and j+y>=0 and j+y<n:
                    value+=kernel[a+x][b+y]*img[i+x][j+y]
                else:
                    value+=0
        value/=ksum
        op[i][j] = value

plt.imshow(op, "gray")
plt.title("Output for mean filter: ")
plt.show()