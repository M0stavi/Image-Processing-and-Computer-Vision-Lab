# -*- coding: utf-8 -*-
"""
Created on Tue May 10 18:43:03 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

def mean(img):
    m = img.shape[0]
    n = img.shape[1]
    freq = np.zeros(256,np.int32)
    for i in range(m):
        for j in range(n):
            pix = int(img[i][j])
            freq[pix]+=1
    pdf = np.zeros(256,np.float32)
    for i in range(256):
        pdf[i] = freq[i]/(m*n)
        
    mean = 0.0
    for i in range(256):
        mean+=pdf[i]*i
    return mean

def std(img):
    m = img.shape[0]
    n = img.shape[1]
    freq = np.zeros(256,np.int32)
    for i in range(m):
        for j in range(n):
            pix = int(img[i][j])
            freq[pix]+=1
    # print(freq)
    pdf = np.zeros(256,np.float32)
    for i in range(256):
        pdf[i] = freq[i]/(m*n)
    # print(pdf)
    mn =mean(img)
    sd = 0.0
    for i in range(256):
        s = (i-mn)*(i-mn)*pdf[i]
        
        sd+=s
    sd=math.sqrt(sd)
    return sd
        
    
    

path = "input2.png"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("INPUT:")

plt.show()

e = 4.0
k0=0.4
k1=0.02
k2=0.4

mni = mean(img)
sd = std(img)

print(mni)
print(sd)

op = img

w_s = int(input("Enter window size: "))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        tem = img[i:i+w_s,j:j+w_s]
        # plt.imshow(tem, 'gray')
        # plt.show()
        mnt = mean(tem)
        sdt = std(tem)
        # print("mn: ",mnt)
        # print(sdt)
        if mnt<=k0*mni and sdt<=k2*sd and sdt>=k1*sd:
            op[i][j] = e*img[i][j]
            # print("hh")

plt.imshow(op,'gray')
plt.title("Output")
plt.show()
        
        




