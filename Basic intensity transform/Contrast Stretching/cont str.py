# -*- coding: utf-8 -*-
"""
Created on Wed May 11 00:57:31 2022

@author: Asus
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = "input.jpg"

img = cv.imread(path)

plt.imshow(img,'gray')

plt.title("Input")

plt.show()

mn = img.min()

mx = img.max()

op = img

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            op[i][j][k] = ((img[i][j][k] - mn)/(mx-mn))*255

plt.imshow(op,'gray')

plt.title("Output")

plt.show()
            