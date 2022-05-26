# -*- coding: utf-8 -*-
"""
Created on Tue May 10 18:44:01 2022

@author: Asus
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

p1 = "F:/Online Class/4-1/zLabs/Vision/lab1/lena.png"

p2 = "F:/Online Class/4-1/zLabs/Vision/ImageProcessing-fromScratchWithOpenCV-/HistogramProcessing/HistogramEqualization/input.jpg"

ref_p = cv.imread(p1)

in_p = cv.imread(p2)

# ref_p = cv.cvtColor(ref_p,cv.COLOR_BGR2GRAY)

# in_p = cv.cvtColor(in_p,cv.COLOR_BGR2GRAY)

ref_p = cv.resize(ref_p,(300,300))

in_p = cv.resize(in_p,(300,300))

plt.imshow(in_p)

plt.show()

plt.hist(ref_p.ravel(),256,(0,256))

plt.title("Reference")

plt.show()

plt.hist(in_p.ravel(),256,(0,256))

plt.title("Input")

plt.show()

op = in_p





for h in range(in_p.shape[0]):
    freq_i = np.zeros(256,np.int32)
    freq_r = np.zeros(256,np.int32)
    
    for i in range(in_p.shape[1]):
        for j in range(in_p.shape[2]):
            pix = int(in_p[h][i][j])
            freq_i[pix]+=1
    pdf_i = np.zeros(256,np.float32)
    
    for i in range(256):
        pdf_i = freq_i/(in_p.shape[1]*in_p.shape[2])
    cdf_i = np.zeros(256,np.float32)
    cdf_i[0] = pdf_i[0]
    for i in range(1,256):
        cdf_i[i] = cdf_i[i-1]+pdf_i[i]
        
    for i in range(ref_p.shape[1]):
        for j in range(ref_p.shape[2]):
            pix = int(ref_p[h][i][j])
            freq_r[pix]+=1
    pdf_r = np.zeros(256,np.float32)
    
    for i in range(256):
        pdf_r = freq_r/(ref_p.shape[1]*ref_p.shape[2])
    cdf_r = np.zeros(256,np.float32)
    cdf_r[0] = pdf_r[0]
    for i in range(1,256):
        cdf_r[i] = cdf_r[i-1]+pdf_r[i]
    for i in range(in_p.shape[1]):
        for j in range(in_p.shape[2]):
            pix = in_p[h][i][j]
            pix = int(pix)
            m = cdf_i[pix]
            
            dis = 1000000.0
            res=pix
            ch=0
            for k in range(256):
                x=cdf_r[k]-m
                if x<0.0:
                    x*=-1.0
                if(x<dis):
                    dis = x
                    res=k
                    ch+=1
            if ch >1:
                m=res
            op[h][i][j] = 255*m
        
    # cdf = cdf_i
    # for i in range(256):
    #     dis = 1000000.0
    #     res=i
    #     ch=0
    #     for j in range(256):
    #         x=cdf_i[i]-cdf_r[j]
    #         if x<0.0:
    #             x*=-1.0
    #         if(x<dis):
    #             dis = x
    #             res=j
    #             ch+=1
    #     if ch >1:
    #         cdf[i]=cdf_r[res]
    # for i in range(in_p.shape[1]):
    #     for j in range(in_p.shape[2]):
    #         pix = in_p[h][i][j]
    #         m=cdf[pix]
    #         op[h][i][j]=255*m
        
            
    
plt.imshow(op)
plt.show()

plt.hist(op.ravel(),256,(0,256))

plt.title("OP")

plt.show()
    
    
    
            
        


