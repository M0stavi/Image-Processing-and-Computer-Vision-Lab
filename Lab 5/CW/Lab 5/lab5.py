# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:55:09 2022

@author: NLP Lab
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = 'sample1.bmp'

img = cv.imread(path,0)

plt.imshow(img,'gray')

plt.show()

t,img = cv.threshold(img,180,255,cv.THRESH_BINARY)

img = img // 255



img = img.astype(np.uint8)

img = 1-img

plt.imshow(img,'gray')

plt.show()

#print(img)
#
#plt.imshow(img,'gray')
#
#plt.show()


kernel = np.ones((3,3),np.uint8)

op = np.zeros((img.shape[0],img.shape[1]), np.uint8)


k=0    

while 1:
    er = cv.erode(img,kernel,iterations = k)
    
#    print(er.shape)
    
    er = er.astype(np.uint8)
    
    s = np.sum(er)
    
#    print(i)
#    
#    print(s)
    
    if s == 0:
        break
#    
    opn = cv.morphologyEx(er,cv.MORPH_OPEN,kernel)
#    
    opn = opn.astype(np.uint8)
#    
    sk = er-opn
#    
#    for i in range(op.shape[0]):
#        for j in range(op.shape[1]):
#            if(sk[i][j] == 1):
#                op[i][j] = 1
    
    op = np.bitwise_or(op,sk)
#    
    k+=1
    
#    print(k)
    
    
plt.imshow(op,'gray')

plt.show()
    
    
    