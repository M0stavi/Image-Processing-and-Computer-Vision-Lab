# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 02:00:04 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

path='hf.jpg'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()