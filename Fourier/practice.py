# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:48:23 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = "C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Fourier/fr.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

# fourier transform

f = np.fft.fft2(img)

m1 = np.abs(f)

plt.imshow(m1,'gray')

plt.show()

shift = np.fft.fftshift(f)

m2 = np.abs(shift)

plt.imshow(m2,'gray')

plt.show()

vis = np.log(shift)

m3 = np.abs(vis)

plt.imshow(m3,'gray')

plt.show()

# inverse fourier transform

op = np.real(np.fft.ifft2(f))

plt.imshow(op,'gray')

plt.show()