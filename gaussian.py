# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 19:46:31 2022

@author: Asus
"""

import cv2 
import matplotlib.pyplot as plt
import numpy as np

# path = 'F:/Online Class/4-1/zLabs/Vision/lab1/lena.png'

# img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# a = kernel.shape[0] // 2
# b = kernel.shape[1] // 2
# n = img.shape[0]
# m = img.shape[1]
# output = np.zeros((n,m), dtype = int)
# for x in range(n):
#     for y in range(m):
#         for s in range(-a, a + 1):
#             for t in range(-b, b + 1):
#                 if x - s < n and x - s >= 0 and y - t < m and y - t >= 0:
#                     output[x][y] += kernel[s + a][t + b] * img[x - s][y - t] # filter function for averaging
#                 else:
#                     output[x][y] += kernel[s + a][t + b] * 0 # we didn't pad the input image with zero explicitly

path = 'F:/Online Class/4-1/zLabs/Vision/lab1/lena.png'

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

plt.show()

kernel = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]), np.float32)

a = kernel.shape[0] // 2

# print(a)

b = kernel.shape[1] // 2

m = img.shape[0]

n = img.shape[1]

result = np.zeros((m,n),dtype="float32")

for i in range(m):
    for j in range(n):
        for x in range(-a, a+1):
            for y in range(-b,b+1):
                if i-x >=0 and i-x < m and j-y >=0 and j-y < n:
                    result[i][j]+= kernel[x+a][y+b] * img[i-x][j-y]
                else:
                    result[i][j] += 0
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()                