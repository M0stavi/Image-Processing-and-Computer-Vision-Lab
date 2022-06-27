# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:53:10 2022

@author: Asus
"""

import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt

path = "F:/Online Class/4-1/zLabs/Vision/lab1/lena.png"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

