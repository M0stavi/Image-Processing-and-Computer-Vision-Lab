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

d0=25
n=1

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        prod = 1
        for k in range(4):
            duv = np.sqrt((i - mm - (xx[k] - mm))**2 + (j - nn - (yy[k] - nn))**2)
            dmuv = np.sqrt((i - mm + (xx[k] - mm))**2 + (j - nn + (yy[k] - nn))**2)
            
            val = (1 / (1 + (d0 / duv) * (2*n))) * (1 / (1 + (d0 / dmuv) * (2*n)))
            prod *= val
        notch[i, j] = prod

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