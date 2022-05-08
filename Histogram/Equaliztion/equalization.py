# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:56:24 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = "C:/Users/Asus/Desktop/hist.jpg" 

img = cv.imread(path)

# print(img.shape[2])

# img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

plt.imshow(img)

plt.title("Input for histogram equalization: ")

plt.show()

plt.hist(img.ravel(),256,(0,256))

plt.title("Histogram of input: ")

plt.show()

op = img

m = int(img.shape[1])
n = img.shape[2]



for h in range(img.shape[0]):
    freq = np.zeros(256,int)
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            pixel = img[h][i][j]
            freq[pixel]+=1
    pdf=np.zeros(256,np.float32)
    
    for i in range(256):
        pdf[i] = freq[i]/(img.shape[1]*img.shape[2])
    cdf = np.zeros(256,np.float32)
    
    cdf[0] = pdf[0]
    
    for i in range(1,256):
        cdf[i] = cdf[i-1]+pdf[i]
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            pix=img[h][i][j]
            m = cdf[pix]
            op[h][i][j] = 255*m

plt.imshow(op)

plt.title("Output for histogram equalization: ")

plt.show()

plt.hist(op.ravel(),256,(0,256))

plt.title("Histogram of output: ")

plt.show()

