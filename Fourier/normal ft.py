# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:59:42 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def scale(img):
    mn = img.min()
    mx = img.max()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = ((img[i][j]-mn)/(mx-mn))*255
    return img.astype(np.float32)

path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Fourier/ein.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Input")

plt.show()

f = np.fft.fft2(img)

mag1 = np.abs(f)

plt.imshow(mag1,'gray')

plt.show()

fshift = np.fft.fftshift(f)

mag = np.abs(fshift)

plt.imshow(mag,'gray')

plt.show()

mag = np.log(mag)

plt.imshow(mag,'gray')

plt.show()

op = np.real(np.fft.ifft2(f))

op = scale(op)

plt.imshow(op,'gray')

plt.show()