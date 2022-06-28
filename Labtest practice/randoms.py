# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:06:05 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = 'rubiks_cube.png'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

sig = 1
s = 2*sig*sig

kernel = np.zeros((5,5),np.float32)

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2

for i in range(-a,a+1):
    for j in range(-b,b+1):
        r = np.exp(-(i*i+j*j)/s)/((2*np.pi)*sig*sig)
        kernel[a+i][b+j] = r
 
for i in range(5):
    print(kernel[i])
    
m = img.shape[0]
n = img.shape[1]


    
op = np.zeros((m,n),np.float32)
    
for i in range(m):
    for j in range(n):
        norm = 0
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    dif = img[i][j]-img[i-x][j-y]
                    r = np.exp(-(dif**2/2*sig**2))*kernel[a+x][b+y]
                    norm+=r
                    op[i][j]+=r*img[i-x][j-y]
        op[i][j]/=norm
        
plt.imshow(op,'gray')
plt.show()
    