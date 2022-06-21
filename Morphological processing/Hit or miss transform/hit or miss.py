# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 00:31:41 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

def erotion(img,kernel):
    m = img.shape[0]
    n = img.shape[1]
    
    a = kernel.shape[0] // 2
    b = kernel.shape[1] // 2
    
    op = np.zeros((m,n),np.int32)
    
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
                
    return op.astype(np.int32)

def dilation(img,kernel):
    m = img.shape[0]
    n = img.shape[1]
    
    a = kernel.shape[0] // 2
    b = kernel.shape[1] // 2
    
    op = np.zeros((m,n),np.int32)
    
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
    return op.astype(np.int32)
path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Morphological processing/Hit or miss transform/in.png"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input")

plt.show()

t, img = cv.threshold(img,100,255,cv.THRESH_BINARY)

img = img // 255

img = img.astype(np.int32)

plt.imshow(img,'gray')

plt.title("Thresh")

plt.show()

s = np.zeros((7,7),np.int32)

s[1:6,2:5] = 1

s[3,1:6] = 1

plt.imshow(s,'gray')

plt.title("Structure")

plt.show()

diam = np.array([[0,1,0],[1,1,1],[0,1,0]],np.int32)

plt.imshow(diam,'gray')

plt.title("Diam")

plt.show()

w = dilation(s, diam)

plt.imshow(w,'gray')

plt.title("W")

plt.show()

y = w-s

img_c = 1-img

z = erotion(img_c,y)

x = erotion(img,s)

plt.imshow(x,'gray')

plt.title("AES")

plt.show()

op = np.bitwise_and(x,z)

plt.imshow(op,'gray')

plt.show()
