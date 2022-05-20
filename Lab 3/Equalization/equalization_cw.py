# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:02:15 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Lab 3/Equalization/in.jpg"

img = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input for equlization")

plt.show()

plt.hist(img.ravel(),256,(0,256))

plt.title("Input histogram")

plt.show()

freq = np.zeros(256,np.int32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pix = int(img[i][j])
        freq[pix]+=1

pdf = np.zeros(256,np.float32)

for i in range(256):
    pdf[i] = freq[i]/(img.shape[0]*img.shape[1])

cdf = np.zeros(256,np.float32)

cdf[0] = pdf[0]

for i in range(1,256):
    cdf[i] = cdf[i-1]+pdf[i]
    
for i in range(256):
    cdf[i] = round(cdf[i]*255.0)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pix = int(round(img[i][j]))
        img[i][j] = cdf[pix]

plt.imshow(img,'gray')

plt.title("Output for equalization")

plt.show()


plt.hist(img.ravel(),256,(0,256))

plt.title("Output histogram")

plt.show()
