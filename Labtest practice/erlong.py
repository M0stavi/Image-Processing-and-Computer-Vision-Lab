# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:53:10 2022

@author: Asus
"""

import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt

def cdf(fn):
    
    cdf = np.zeros(256,np.float32)
    pdf = np.zeros(256,np.float32)
    
    for i in range(256):
        pdf = fn[i]/fn.sum()
    cdf[0] = pdf[0]
    for i in range(1,256):
        cdf[i] = cdf[i-1]+pdf[i]
    
    return cdf

def cdfi(img):
    freq = np.zeros(256,np.float32)
    cdf = np.zeros(256,np.float32)
    pdf = np.zeros(256,np.float32)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pix= np.round(img[i][j])
            freq[pix] += 1
    
    for i in range(256):
        pdf[i] = freq[i]/(img.shape[0]*img.shape[1])
        
    cdf[0] = pdf[0]
    
    for i in range(1,256):
        cdf[i] = cdf[i-1]+pdf[i]
        
    return cdf
    

path = "F:/Online Class/4-1/zLabs/Vision/lab1/lena.png"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

freq = np.zeros(256,np.float32)
cdf = np.zeros(256,np.float32)
pdf = np.zeros(256,np.float32)

m = img.shape[0]
n = img.shape[1]

cdf = cdfi(img)

er = np.zeros(256,np.float32)

k = 9

m1= 1

for i in range(256):
    r = (i**(k-1)*np.exp(-(i/m1)))/(m1**k*math.factorial(k-1))
    er[i] = r
    
plt.plot(er)
plt.show()

cdfe = np.zeros(256,np.float32)

cdfe[0] = er[0]

for i in range(1,256):
    cdfe[i] = cdfe[i-1]+er[i]
    
plt.plot(cdfe)

plt.show()

cdfe = np.round(255*cdfe)
cdf = np.round(cdf*255)

for i in range(256):
    dis = 1000
    res = i
    for j in range(256):
        m = np.abs(cdf[i]-cdfe[j])
        if m<dis:
            dis = m
            res = j
    cdf[i] = res
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pix = np.round(img[i][j])
        # m = cdf[pix]
        img[i][j] = cdf[pix]
        
plt.hist(img.ravel(),256,(0,256))

plt.show()