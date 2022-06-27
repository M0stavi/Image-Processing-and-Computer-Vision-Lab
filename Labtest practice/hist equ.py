# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 20:08:21 2022

@author: Asus
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

path = 'in.jpg'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

plt.hist(img.ravel(),256,(0,256))

plt.show()

freq = np.zeros(256,np.float32)

m = img.shape[0]
n = img.shape[1]

for i in range(m):
    for j in range(n):
        pix = int(np.round(img[i][j]))
        freq[pix] += 1

pdf = np.zeros(256,np.float32)

for i in range(256):
    pdf[i] = freq[i]/(m*n)
    
cdf = np.zeros(256,np.float32)

cdf[0] = pdf[0]

for i in range(1,256):
    cdf[i] = cdf[i-1]+pdf[i]
    
plt.plot(cdf)

plt.show()

cdf = np.round(cdf*255.0)

for i in range(m):
    for j in range(n):
        pix = int(np.round(img[i][j]))
        img[i][j] = cdf[pix]

plt.hist(img.ravel(),256,(0,256))

plt.show()

plt.imshow(img,'gray')

plt.show()
        