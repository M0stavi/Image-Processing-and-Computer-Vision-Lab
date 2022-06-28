# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 01:24:44 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

xx= []
yy=[]

def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        xx.append(y)
        yy.append(x)
        
        
path = 'pi.jpg'
img = cv.imread(path)


img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()


f = np.fft.fft2(img)

sft = np.fft.fftshift(f)

mag = np.abs(sft)

mag=np.log(mag)

plt.imshow(mag,'gray')

plt.show()

k=0

while 1:
    if k==2:
        break
    mag = np.log(mag)
    k+=1
    
cv.imshow('image',mag)
    

cv.setMouseCallback('image',click_event)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imshow('image',mag)
    

cv.setMouseCallback('image',click_event)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imshow('image',mag)
    

cv.setMouseCallback('image',click_event)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imshow('image',mag)
    

cv.setMouseCallback('image',click_event)

cv.waitKey(0)

cv.destroyAllWindows()

mm = img.shape[0]//2
nn = img.shape[1]//2

notch = np.zeros((img.shape[0],img.shape[1]),np.float32)

d0=50
n=2

for u in range(img.shape[0]):
    for v in range(img.shape[1]):
        for k in range(4):
            dk = np.sqrt((u-mm-xx[k])**2+(v-nn-yy[k])**2)
            dkk = np.sqrt((u-mm+xx[k])**2+(v-nn+yy[k])**2)
            if dk>d0 or dkk>d0:
                if dk==0:
                    dk=1
                if dkk == 0:
                    dkk = 1
                notch[u][v] += (1/(1+(d0/dk)**(2*n)))+(1/(1+(d0/dkk)**(2*n)))

ni = np.fft.ifftshift(notch)
# ni = np.fft.fftshift(ni)
plt.imshow(ni,'gray')
plt.show()
plt.imshow(notch,'gray')
plt.show()

mag = np.abs(sft)

phase=np.angle(sft)

mag=mag*ni

op = np.multiply(mag,np.exp(1j*phase))

op=np.fft.ifftshift(op)

op = np.real(np.fft.ifft2(op))

plt.imshow(op,'gray')
plt.show()