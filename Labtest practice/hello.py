# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 08:29:01 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = 'lena.png'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

kernel = np.array(([1,0],[0,-1]),np.float32)
    
m = img.shape[0]
n = img.shape[1]

a = kernel.shape[0]//2
b = a


op = np.zeros((m,n),np.float32)

for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y< n and x+a>=0 and x+a<kernel.shape[0] and y+b>=0 and y+b<kernel.shape[0] :
                    op[i][j]+=img[i-x][j-y]*kernel[x+a][y+b]

# op = cv.filter2D(img,dde)
                    
plt.imshow(op,'gray')

plt.show()