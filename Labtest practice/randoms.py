# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:06:05 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = "hf.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

t,img = cv.threshold(img,180,255,cv.THRESH_BINARY)

img = img//255

plt.imshow(img,'gray')

plt.show()

