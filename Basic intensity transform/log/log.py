# -*- coding: utf-8 -*-
"""
Created on Wed May 11 01:14:32 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = "input.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input:")

plt.show()

op = img

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        c = 1.0
        x = c*math.log(1+img[i][j])
        l = img.max()
        r = img.min()
        mn = c*math.log(1+l)
        mx = c*math.log(1+r)
        op[i][j] = (((r-l)*(x-mn))/(mx-mn)) + l
plt.imshow(op,'gray')
plt.title("Output")
plt.show()        