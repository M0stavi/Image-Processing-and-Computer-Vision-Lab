# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 08:29:01 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

xx = []
yy = []

def click_event(event,x,y,a,b):
    if event == cv.EVENT_LBUTTONDOWN:
        xx.append(y)
        yy.append(x)

path = 'pi.jpg'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

f = np.fft.fft2(img)

sft = np.fft.fftshift(f)

magx = np.abs(sft)

k = 0

while 1:
    if k == 3:
        break
    magx = np.log(magx)
    k+=1
    
cv.imshow('image',magx)

cv.setMouseCallback('image',click_event)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imshow('image',magx)

cv.setMouseCallback('image',click_event)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imshow('image',magx)

cv.setMouseCallback('image',click_event)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imshow('image',magx)

cv.setMouseCallback('image',click_event)

cv.waitKey(0)

cv.destroyAllWindows()

m = img.shape[0]
n = img.shape[1]

d0=25
nn=1

notch = np.zeros((m,n),np.float32)

# for u in range(img.shape[0]):
#     for v in range(img.shape[1]):
#         prod=1
#         for k in range(4):
#             d= np.sqrt((u-m//2-(xx[k]-m//2))**2+(v-n//2-(yy[k]-n//2))**2)
            
#             dk= np.sqrt((u-m//2+(xx[k]-m//2))**2+(v-n//2+(yy[k]-n//2))**2)
            
#             # if  dk<=0:
#             #     dk=1
#             # if d<=0:
#             #     d=1
            
#             prod*=1/ ( ( 1+(d0/d)**(2*n)  )*( 1+(d0/dk)**(2*n) ) )
#         notch[u][v] = prod

print(m,' ',n)
for u in range(m):
    for v in range(n):
        prod = 1
        for k in range(4):
            d = np.sqrt((u-m//2-(xx[k]-m//2))**2+(v-n//2-(yy[k]-n//2))**2)
            dk = np.sqrt((u-m//2+(xx[k]-m//2))**2+(v-n//2+(yy[k]-n//2))**2)
            prod*=1/ ( ( 1+(d0/d)*(2*nn)  )*( 1+(d0/dk)*(2*nn) ) )
        notch[u][v] = prod
        
        # print(1)

mag = np.abs(sft)

phase = np.angle(sft)

mag*=notch

op = np.multiply(mag,np.exp(1j*phase))

op = np.fft.ifftshift(op)

op = np.real(np.fft.fft2(op))
        
plt.imshow(notch,'gray')
plt.show()

op = cv.rotate(op, cv.ROTATE_180)

plt.imshow(op,'gray')
plt.show()