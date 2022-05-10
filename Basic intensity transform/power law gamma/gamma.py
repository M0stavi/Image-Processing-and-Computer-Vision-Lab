# -*- coding: utf-8 -*-
"""
Created on Wed May 11 01:31:34 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = "input1.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

img = cv.resize(img,(200,200))

plt.imshow(img,'gray')

plt.title("Input:")

plt.show()

op = np.zeros((img.shape[0],img.shape[1]),np.float32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        c = 1.0
        g = 0.5
        
        x = c*math.pow(img[i][j],g)
        l = img.min()
        r = img.max()
        mn = c*math.pow(l,g)
        mx = c*math.pow(r,g)
        op[i][j] = (((r-l)*(x-mn))/(mx-mn)) + l
plt.imshow(op,'gray')
plt.title("Output")
plt.show()        