# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:06:05 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = "F:/Online Class/4-1/zLabs/Vision/lab1/lena.png"

img = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

g1 = np.zeros(256,np.float32)

g2 = np.zeros(256,np.float32)

freq = np.zeros(256,np.float32)
pdf = np.zeros(256,np.float32)
cdf =np.zeros(256,np.float32)

m1 = 80
sd1= 25

m2 = 200
sd2 = 15

for i in range(256):
    r = np.exp(-(((i-m1)**2)/(2*sd1**2)))/(sd1*np.sqrt(2*np.pi))
    g1[i] = r
    
plt.plot(g1)
plt.show()

for i in range(256):
    r = np.exp(-(((i-m2)**2)/(2*sd2**2)))/(sd2*np.sqrt(2*np.pi))
    g2[i] = r
    
plt.plot(g2)
plt.show()

g = g1+g2

for i in range(256):
    pdf[i] = g[i]/g.sum()
    
cdf[0] = pdf[0]

for i in range(1,256):
    cdf[i] = cdf[i-1]+pdf[i]
    
cdfg = cdf

pdf = np.zeros(256,np.float32)
cdf =np.zeros(256,np.float32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pix= np.round(img[i][j])
        freq[pix] += 1
        
plt.hist(img.ravel(),256,(0,256))
plt.show()

for i in range(256):
    pdf[i] = freq[i]/(img.shape[0]*img.shape[1])
    
cdf[0] = pdf[0]

for i in range(1,256):
    cdf[i]  = cdf[i-1]+pdf[i]
    
cdf = np.round(cdf*255)
cdfg = np.round(cdfg*255)  
for i in range(256):
    m = cdf[i]
    dis = 1000000.0
    res = i
    for j in range(256):
        x = np.abs(cdfg[j] - m)
        
        if x<dis:
            dis= x
            res=j
    
    cdf[i] = res
    


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pix = np.round(img[i][j])
        img[i][j] = cdf[pix]
        
plt.hist(img.ravel(),256,(0,256))

plt.show()
        