# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:19:45 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

path = "C:/Users/Asus/Desktop/homo.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


plt.imshow(img,'gray')

plt.title("Input image:")

plt.show()

img = np.log1p(img)

filt = np.zeros((img.shape[0],img.shape[1]), np.float32)

gh = 1.2
gl = 0.5
c = 0.1
d0 = 50

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        u = (i-img.shape[0]/2)**2
        v = (j-img.shape[1]/2)**2
        r = math.exp(-c*((u+v)/d0**2))
        r = (gh-gl)*(1-r)+gl
        filt[i][j] = r

plt.imshow(filt,'gray')
plt.title("Homomorphic filter:")
plt.show()

# f = cv.dft(np.float32(img),flags=cv.DFT_COMPLEX_OUTPUT)

# magnitude = np.log1p(cv.magnitude(dft[:,:,0],dft[:,:,1]))

f = np.fft.fft2(img)

shift = np.fft.fftshift(f)

mag = np.abs(shift)

plt.imshow(np.log(mag),'gray')

plt.title("Magnitude before multiplication with filter:")

plt.show()

phase = np.angle(shift)

mag = mag*filt

plt.imshow(np.log(mag),'gray')

plt.title("Magnitude after multiplication with filter:")

plt.show()

op = np.multiply(mag,np.exp(1j*phase))

ishift = np.fft.ifftshift(op)

inv = np.real(np.fft.ifft2(ishift))

inv = np.expm1(inv)

plt.imshow(inv,'gray')

plt.title("Output image:")

plt.show()