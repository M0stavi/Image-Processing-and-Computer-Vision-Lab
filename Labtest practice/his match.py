# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 20:52:57 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = "F:/Online Class/4-1/zLabs/Vision/lab1/lena.png"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

plt.hist(img.ravel(),256,(0,256))

plt.show()

g_a = np.zeros(256,np.float32)

g_b = np.zeros(256,np.float32)

m_a = 130

sd_a = 35

m_b = 200

sd_b = 20

freq = np.zeros(256,np.float32)
pdf = np.zeros(256,np.float32)
cdf = np.zeros(256,np.float32)

m = img.shape[0]
n = img.shape[1]

for i in range(m):
    for j in range(n):
        pix = int(np.round(img[i][j]))
        freq[pix] += 1

for i in range(256):
    pdf[i] = freq[i]/(m*n)

cdf[0] = pdf[0]

for i in range(1,256):
    cdf[i] = cdf[i-1]+pdf[i]

cdf_im = cdf

freq = np.zeros(256,np.float32)
pdf = np.zeros(256,np.float32)
cdf = np.zeros(256,np.float32)

for i in range(256):
    r = (i-m_a)
    # r/=sd_a
    r = r*r
    r/=(sd_a**2)
    # r/=2
    r=np.exp(-(r))
    r/=(np.sqrt(2*np.pi)*sd_a)
    g_a[i] = r

plt.plot(g_a)
plt.show()


for i in range(256):
    r = (i-m_b)
    # r/=sd_b
    r = r*r
    # r/=2
    r/=(sd_b**2)
    r=np.exp(-(r))
    r/=(np.sqrt(2*np.pi)*sd_b)
    g_b[i] = r

plt.plot(g_b)
plt.show()

g = g_a+g_b

freq = g

for i in range(256):
    pdf[i] = freq[i]/(freq.sum())
    
cdf[0] = pdf[0]

for i in range(1,256):
    cdf[i] = cdf[i-1]+pdf[i]
    
cdf_g = cdf

cdf_im = np.round(cdf_im*255.0)

cdf_g = np.round(cdf_g*255.0)

for i in range(256):
    m = cdf_im[i]
    res = 0
    dis  = 1000000.0
    for j in range(256):
        # m = cdf_img[i]
        f = m-cdf_g[j]
        if f<0:
            f*=-1
        if f<dis:
            res = j
            dis = f
    cdf[i] = res
m = img.shape[0]
n = img.shape[1]
for i in range(m):
    for j in range(n):
        pix = np.round(img[i][j])
        
        m = cdf[pix]
        img[i][j] = m
 
# for i in range(m):
#     for j in range(n):
#         pix = np.round(img[i][j])
        
#         m = cdf_im[pix]
        
#         dis  =1000000.0
#         res = pix
#         for k in range(256):
#             a = cdf_g[k]
#             f= m-a
#             if f<0:
#                 f*=-1
#             if f<dis:
#                 res = k
#                 dis=f
#         img[i][j] = res

plt.hist(img.ravel(),256,(0,256))

plt.show()