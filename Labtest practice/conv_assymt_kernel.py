# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 22:42:11 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

kernel = np.array(([1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]),np.float32)

a = kernel.shape[0]//2
b = kernel.shape[1]//2
b-=1

for i in range(-a,a):
    for j in range(-b,b+2):
        print(kernel[a+i][b+j])
        
img = np.array(([1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]),np.float32)

i=1
j=2
2
print(img[i][j])

for x in range(-a,a):
    for y in range(-b,b+2):
        print(kernel[a+x][b+y],' ',img[i-x][j-y])
        