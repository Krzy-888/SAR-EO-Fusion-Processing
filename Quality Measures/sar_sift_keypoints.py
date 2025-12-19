# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:34:36 2017

@author: Administrator #EDITED BY ME, STILL WORKING ON IT
"""
import __future__
import cv2
import time
import numpy as np

import build_scale
import find_scale_extreme
import calc_descriptors
import match
import display
import ransac
import image_fusion
import MI
import matplotlib.pyplot as plt
import RMSE

#read image

img = cv2.imread(r"Norm/CAPELLA_C05_URRC_Subset1000_2_gray.png",0) # image to be registered
#rgb2gray
start = time.time()
gray = img/255.
fitst = time.time() - start
print(f"1/3 {fitst}")
#initial parameter
sigma = 2                      #initial layer scale
ratio = 2**(1/3.)               #scale ratio
Mmax = 8                       #layer number
d = 0.04
d_SH_1 = 0.8                   #Harros function threshold
distRatio = 0.9
error_threshold = 1

#Creat sar-harris function
sar_harris_function_,gradient_,angle_ = build_scale.build_scale(gray,sigma,Mmax,ratio,d)
fitst = time.time() - start
print(f"2/3 {fitst}")
#Feaarure point detection
GR_key_array_ = find_scale_extreme.find_scale_extreme(sar_harris_function_,d_SH_1,sigma,ratio,gradient_,angle_)
fitst = time.time() - start
print(f"3/3 {fitst}")

print(len(GR_key_array_[:,0:2]) )
