# -*- coding: utf-8 -*-
"""
Created on Wed May  4 12:52:51 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

def clip(img):
    m = img.shape[0]
    n = img.shape[1]
    for i in range(m):
        for j in range(n):
            if img[i][j] > 255:
                img[i][j] = 255
            if img[i][j] < 0:
                img[i][j] = 0
    return img.astype(np.float32)

def scale(img):
    g_m = img-img.min()
    g_s = 255*(g_m/g_m.max())
    return g_s.astype(np.float32)

def Zero_crossing(image):
    z_c_image = np.zeros(image.shape)
    
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1


            # If both negative and positive values exist in 
            # the pixel neighborhood, then that pixel is a 
            # potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood
            # difference with the pixel

            if z_c:
                if image[i,j]>0:
                    z_c_image[i, j] = image[i,j] + np.abs(e)
                elif image[i,j]<0:
                    z_c_image[i, j] = np.abs(image[i,j]) + d
                
    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image/z_c_image.max()*255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image.astype(np.float32)

path = "F:/Online Class/4-1/zLabs/Vision/lab1/lena.png"

img = cv.imread(path)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.title("Inpur for laplace raw")

plt.show()











im_g = img

blur = cv.GaussianBlur(im_g,(3,3),0)

laplacian = cv.Laplacian(blur,cv.CV_64F)

# But this tends to localize the edge towards the brighter side.
# laplacian1 = laplacian/laplacian.max()

laplacian1 = Zero_crossing(laplacian)


plt.imshow(laplacian,'gray')

plt.title("Function: ")

plt.show()

plt.imshow(laplacian1,'gray')

plt.title("Function op: ")

plt.show()

print("Enter kernel height: ")

k_h = int(input())

print("Enter kernel weidth: ")

k_w = int(input())

sigma = 1.0

s = sigma*sigma

pi = 3.1416

kernel = np.zeros((k_h,k_w), np.float32)

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2

m = img.shape[0]
n = img.shape[1]

op = np.zeros((m,n), np.float32)

for x in range(-a,a+1):
    for y in range(-b,b+1):
        sqsum = (x*x+y*y)
        term = (sqsum)/(2*s)
        r = math.exp(-(term))
        term = 1-term
        r = r*term*(-1/(pi*s*s))
        kernel[a+x][b+y] = r
        
# kernel[1][1]*=-1

for i in range(k_h):
    print(kernel[i])
    
for i in range(m):
    for j in range(n):
        for x in range(-a,a+1):
            for y in range(-b,b+1):
                if i-x>=0 and i-x<m and j-y>=0 and j-y<n:
                    op[i][j]+=kernel[x+a][y+b]*img[i-x][j-y]
                else:
                    op[i][j]+=0

plt.imshow(op,'gray')

plt.title("Laplace filtered: ")

plt.show()

op_c = clip(op)

plt.imshow(op_c,'gray')

plt.title("Clipped: ")

plt.show()

op_s = scale(op)

plt.imshow(op_s,'gray')

plt.title("Scaled: ")

plt.show()

img = img-op_c #center of kernel is negative

img = clip(img)

plt.imshow(img,'gray')

plt.title("After enhancing: ")

plt.show()





        