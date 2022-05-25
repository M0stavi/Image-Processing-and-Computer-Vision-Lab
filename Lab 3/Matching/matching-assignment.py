# -*- coding: utf-8 -*-
"""
Created on Fri May 20 21:35:21 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

g_a = np.zeros(256,np.float32)

m_a = 80.0

sd_a = 35.0

for i in range(256):
    r = i-m_a
    r = r*r
    r = r/(sd_a*sd_a)
    r = math.exp(-(r))
    r = r/(sd_a*math.sqrt(2*3.1416))
    g_a[i] = r
    
plt.plot(g_a)

plt.title("Gaussian function 1")

plt.show()

g_b = np.zeros(256,np.float32)

m_b = 200.0

sd_b = 20.0

for i in range(256):
    r = i-m_b
    r = r*r
    r = r/(sd_b*sd_b)
    r = math.exp(-(r))
    r = r/(sd_b*math.sqrt(2*3.1416))
    g_b[i] = r
    
plt.plot(g_b)

plt.title("Gaussian function 2")

plt.show()

gauss = g_a+g_b

plt.plot(gauss)

plt.title("Added Gaussian function")

plt.show()

fr_g = gauss

t_fr = gauss.sum()

pdf_g = np.zeros(256,np.float32)

for i in range(256):
    pdf_g[i] = fr_g[i]/t_fr

cdf_g = np.zeros(256,np.float32)

cdf_g[0] = 0.0

for i in range(1,255):
    cdf_g[i] = cdf_g[i-1]+pdf_g[i]
    

for i in range(256):
    cdf_g[i] = round(cdf_g[i]*255)

path = "F:/Online Class/4-1/zLabs/Vision/lab1/lena.png"

img = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input image for matching")

plt.show()

plt.hist(img.ravel(),256,(0,256))

plt.title("Input image histogram")

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
        pix = int(img[i][j])
        m = cdf[pix]
            
        dis = 10000
        res=pix
        # ch=0
        for k in range(256):
            x=cdf_g[k]-m
            if x<0:
                x*=-1
            if(x<dis):
                dis = x
                res=k
        
        m=res
        
        img[i][j] = m

plt.imshow(img,'gray')

plt.title("Output for matching")

plt.show()


plt.hist(img.ravel(),256,(0,256))

plt.title("Output histogram")

plt.show()

freq_op = np.zeros(256)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pix = int(img[i][j])
        freq_op[pix]+=1

plt.plot(freq_op)

plt.title("Plot of intensity frequencies of output image:")

plt.show()


