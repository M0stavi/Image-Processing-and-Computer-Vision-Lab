# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 23:08:45 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/Users/Asus/imagelab/Image-Processing-and-Computer-Vision-Lab/Lab 4/assignment/assm.png'

fxy = cv.imread(path)

fxy = cv.cvtColor(fxy, cv.COLOR_BGR2GRAY)

plt.imshow(fxy,'gray')

plt.title("Input for motion blur")

plt.show()

k = np.zeros((15,15),np.float32)

for i in range(15):
    k[i][i] = 1
    
    
print(k)



m = fxy.shape[0]
n = fxy.shape[1]

a = int(fxy.shape[0]//2 - k.shape[0]//2)

k_pad = np.pad(k, (a,a-1), 'constant', constant_values=(0))

print(k_pad.shape[0], k_pad.shape[1])

fxy = cv.resize(fxy,(k_pad.shape[0],k_pad.shape[0]))

# fourier transform

F = np.fft.fft2(fxy)

H = np.fft.fft2(k_pad)

f_show = np.fft.fftshift(np.log(np.abs(F)+1))

h_show  = np.fft.fftshift(np.log(np.abs(H)+1))

plt.imshow(f_show, "gray")

plt.title("Image magnitude")

plt.show()

plt.imshow(h_show, "gray")

plt.title("Filter magnitude")

plt.show()

G = np.multiply(F,H)

g_show = np.fft.fftshift(np.log(np.abs(G)+1))

plt.imshow(g_show, "gray")

plt.title("Blur magnitude")

plt.show()

x = np.real(np.fft.ifft2(G))

x = np.fft.fftshift(x)

plt.imshow(x,'gray')

plt.title("Blurred Image:")

plt.show()

g = np.fft.fftshift(G)

g = np.real(g)

F_hat = np.divide(G,H)

f_hat = np.real(np.fft.ifft2(F_hat))

plt.imshow(f_hat,'gray')

plt.title("Output of motion deblur")

plt.show()