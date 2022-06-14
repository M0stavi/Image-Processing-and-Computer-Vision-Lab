# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 19:59:49 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

img = np.zeros((5,6),np.float32)

for i in range(6):
    img[0][i] = 1

for i in range(6):
    img[4][i] = 1
    
img[1][1] = 1

img[1][4] = 1

img[3][1] = 1

img[3][4] = 1

for i in range(5):
    img[i][0] = 1
    
for i in range(5):
    img[i][5] = 1

print(img)

s = np.ones((3,1),np.float32)

m1 = img.shape[0]
n1 = img.shape[1]

a = s.shape[0]//2
b = s.shape[1] // 2

img = cv.copyMakeBorder(img, a, a, b, b, cv.BORDER_CONSTANT, (0,0,0))

op = np.zeros((img.shape[0],img.shape[1]), np.float32)

m = img.shape[0]
n = img.shape[1]

for i in range(m):
    for j in range(n):
        flag = 1
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if x+i >= 0 and x+i <m and j+y>=0 and j+y<n:
                    if img[i+x][j+y] != s[a+x][b+y]:
                        flag = 0
        
        if flag:
            op[i][j] = 1
        else:
            op[i][j] = 0
            
opf = np.zeros((m1,n1),np.float32)

for i in range(a,m-a):
    for j in range(b,n-b):
        opf[i-a][j-b] = op[i][j]

print(opf)