# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:31:36 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path = "C:/Users/Asus/Desktop/local.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

k_s = int(input("Enter kernel size: "))

kernel = np.ones((k_s,k_s), np.float32)

plt.imshow(img,'gray')

plt.title("Input for local histogram: ")

plt.show()

plt.hist(img.ravel(),256,(0,256))

plt.show()

m = img.shape[0]
n = img.shape[1]

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2

# op=np.zeros((img.shape[0],img.shape[1]),np.float32)

op=img

for ii in range(img.shape[0]):
    for jj in range(img.shape[1]):
        tem = img[ii:ii+k_s,jj:jj+k_s]
        # for x in range(-a,a+1):
        #     for y in range(-b,b+1):
        #         if x+ii>=0 and x+ii<m and jj+y>=0 and jj+y<n:
        #             tem[x+a][b+y] = img[ii+x][jj+y]
        # cdf = np.zeros(256,np.float32)
        
        # for i in range(tem.shape[0]):
        #     print(tem[i])
        # plt.imshow(tem,'gray')
        # plt.show()
        freq = np.zeros(256,np.int32)
        for i in range(tem.shape[0]):
            for j in range(tem.shape[1]):
                pix = int(tem[i][j])
                # print("PIX: ", pix)
                freq[pix]+=1
        # print(freq)
        pdf = np.zeros(256,np.float32)
        for i in range(256):
            pdf[i] = freq[i]/(tem.shape[0]*tem.shape[1])
        # plt.hist(pdf.ravel(),0,(0,256))
        cdf = np.zeros(256,np.float32)
        cdf[0]=pdf[0]
        for i in range(1,256):
            cdf[i] = cdf[i-1]+pdf[i]
        # print(cdf)
        pix = int(img[ii][jj])
        m=cdf[pix]
        # print("M", m)
        op[ii][jj] = 255*m
                
# for i in range(op.shape[0]):
#     print(op[i])
        

plt.imshow(op,'gray')
plt.show()
            
                    
        