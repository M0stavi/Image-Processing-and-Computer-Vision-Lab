# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 02:37:27 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def erotion(img,kernel):
    flag = 1
    m = img.shape[0]
    n = img.shape[1]
    
    a = kernel.shape[0] // 2
    b = kernel.shape[1] // 2
    
    op = np.zeros((m,n),np.int32)
    
    for i in range(m):
        for j in range(n):
            
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
    def ddd(img,se):
        flag_d = 0
        out_d = np.zeros_like(img)
        img_r = img.shape[0]
        img_c = img.shape[1]
        se_r = se.shape[0]
        pad = math.floor(se_r/2)
        for i in range(img_r-math.ceil(se_r/2)):
            for j in range(img_c-math.ceil(se_r/2)):
                for x in range(se_r):
                    for y in range(se_r):
                        if i+x>=0 and i+x<img_r and j+y>=0 and j+y<img_c:
                            if img[i+x][j+y] == se[x][y] and img[i+x][j+y] == 1:
                                flag_d = 1
                if(flag_d):
                    out_d[i+pad][j+pad] = 1
                else:
                    out_d[i+pad][j+pad] = 0
                flag_d = 0
        return out_d.astype(np.int32)

img = np.array([[0,0,0,0,0,0,0,0],
                [0,0,0,1,1,1,0,0],
                [0,0,1,0,0,1,0,0],
                [0,1,0,0,0,1,0,0],
                [0,0,1,0,0,1,0,0],
                [0,0,1,0,0,1,0,0],
                [0,0,1,0,0,1,0,0],
                [0,0,1,0,0,1,0,0],
                [0,1,0,0,0,1,0,0],
                [1,1,1,1,1,1,1,0]])

plt.imshow(img,'gray')

plt.axis('off')

plt.title("Input")

plt.show()

op = np.zeros_like(img)

com = 1-img

se = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]])

while 1:
    op1 = op
    a = dilation(op,se)
    op = np.bitwise_and(a.astype(,com)
    if np.array_equal(op, op1):
        break
    
hole = np.bitwise_or(img,op)
print(hole)
plt.imshow(hole,'gray')
plt.show() 
    
