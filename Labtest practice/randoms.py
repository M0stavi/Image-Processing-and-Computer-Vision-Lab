# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:06:05 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Lab 5/CW/Lab 5/sample1.bmp"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()    

t,img = cv.threshold(img,180,255,cv.THRESH_BINARY)

img = img//255

img = img.astype(np.uint8)

img = 1-img

plt.imshow(img,'gray')

plt.show()  

kernel = np.ones((3,3),np.uint8)

op = np.zeros((img.shape[0],img.shape[1]),np.uint8)
k=0
while(1):
    er = cv.erode(img,kernel,iterations = k)
    s = er.sum()
    if s==0:
        break
    er = er.astype(np.uint8)
    
    opn = cv.morphologyEx(er,cv.MORPH_OPEN,kernel)
    
    opn = opn.astype(np.uint8)
    
    sk = er-opn
    
    sk = sk.astype(np.uint8)
    
    op = np.bitwise_or(op,sk)
    k+=1
    
plt.imshow(op, 'gray')

plt.show()