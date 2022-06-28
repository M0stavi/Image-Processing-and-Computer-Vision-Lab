# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:06:05 2022

@author: Asus
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('pi.jpg', 0)

plt.imshow(img, cmap='gray')
plt.title('Input')
plt.show()

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
magnitude = np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))

plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude')
plt.show()

dft_shift = np.fft.fftshift(dft)
magnitude_specturm = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.imshow(magnitude_specturm, cmap='gray')
plt.title('Magnitude Specturm after Shift')
plt.show()

d0 = 25
n = 1

butter_filter = np.ones(img.shape)

center_i, center_j = img.shape[0] // 2, img.shape[1] // 2

u = [380, 235, 155, 310]
v = [277, 470, 400, 400]

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        prod = 1
        for k in range(4):
            duv = np.sqrt((i - center_i - (u[k] - center_i))**2 + (j - center_j - (v[k] - center_j))**2)
            dmuv = np.sqrt((i - center_i + (u[k] - center_i))**2 + (j - center_i + (v[k] - center_j))**2)
            
            val = (1 / (1 + (d0 / duv) * (2*n))) * (1 / (1 + (d0 / dmuv) * (2*n)))
            prod *= val
        butter_filter[i, j] = prod
        
plt.imshow(butter_filter, cmap='gray')
plt.title('Notch Filer')
plt.show()

dft_shift[:, :, 0] = dft_shift[:, :, 0] * butter_filter

f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back1 = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.imshow(img_back1, cmap='gray')
plt.title('Output')
plt.show()