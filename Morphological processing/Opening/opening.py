# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:22:17 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

def erotion(img, kernel):
    m = img.shape[0]
    n = img.shape[1]
    
    a = kernel.shape[0] // 2
    b = kernel.shape[1] // 2
    
    op = np.zeros((m,n), np.float32)
    
    for i in range(m):
        for j in range(n):
            flag = 1
            for x in range(-a,a+1):
                for y in range(-b,b+1):
                    if i+x>=0 and i+x<m and j+y>=0 and j+y<n:
                        if img[i+x][j+y] != kernel[a+x][b+y]:
                            flag=0
            if flag:
                op[i][j] = 1
            else:
                op[i][j] = 0
    return op.astype(np.float32)

def dilation(img,kernel):
    m = img.shape[0]
    n = img.shape[1]
    
    a = kernel.shape[0] // 2
    b = kernel.shape[1] // 2
    
    op = np.zeros((m,n),np.float32)
    
    for i in range(m):
        for j in range(n):
            flag = 0
            for x in range(-a,a+1):
                for y in range(-b,b+1):
                    if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                        if img[i-x][j-y] == kernel[a+x][b+y]:
                            flag=1
            if flag:
                op[i][j] = 1
            else:
                op[i][j] = 0
    return op.astype(np.float32)
    
    
    

path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Morphological processing/Opening/in.png" 

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input for opening:")

plt.show()

t, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)

img = img // 255

k_h = int(input("Enter kernel height: "))

k_w = int(input("Enter kernel weidth: "))

kernel = np.ones((k_h,k_w),np.float32)

op = erotion(img,kernel)

plt.imshow(op,'gray')

plt.title("Erotion:")

plt.show()

op = dilation(op, kernel)

plt.imshow(op,'gray')

plt.title("Dilation followed by erotion:")

plt.show()