# -*- coding: utf-8 -*-
"""
Created on Wed May 11 02:34:18 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy

path = "in.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("In")

plt.show()

imgs = [255 * ((img& (1<<i)) >>i) for i in range(8)]

for i in range(8):
    plt.imshow(imgs[i],'gray')

    plt.title("Out")

    plt.show()
    