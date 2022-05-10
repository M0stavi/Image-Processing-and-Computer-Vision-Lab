# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:14:55 2022

@author: Asus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

path = "F:/Online Class/4-1/zLabs/Vision/ImageProcessing-fromScratchWithOpenCV-/HistogramProcessing/HistogramEqualization/input.jpg"

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

op = ((img-img.min())/(img.max()-img.min()))*255

plt.imshow(op,'gray')

plt.show()