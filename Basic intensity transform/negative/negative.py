# -*- coding: utf-8 -*-
"""
Created on Wed May 11 01:58:28 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = "input.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# img = cv.resize(img,(200,200))

plt.imshow(img,'gray')

plt.title("Input:")

plt.show()

op = img

mx=img.max()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        op[i][j] = mx-1-img[i][j]

plt.imshow(op,'gray')
plt.title("Output")
plt.show()        