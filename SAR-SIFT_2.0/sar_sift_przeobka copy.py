# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:34:36 2017

@author: Administrator
"""
import __future__
import cv2
import time
import numpy as np

import build_scale
import find_scale_extreme
import matplotlib.pyplot as plt
import match_copy
import calc_descriptors_copy
#read image
img1 = cv2.imread(r"Norm/CAPELLA_C05_URRC_Subset1000_2_gray.png",0) # image to be registered

print("img readed")
#rgb2gray
gray1 = img1/255.
print("img normalized")

#initial parameter
time1 = time.time()
sigma = 2                      #initial layer scale
ratio = 2**(1/3.)               #scale ratio
Mmax = 8                       #layer number
d = 0.04
d_SH_1 = 0.8                   #Harros function threshold
d_SH_2 = 0.8                   #Harros function threshold
distRatio = 0.9
error_threshold = 3

#Creat sar-harris function
print("sar_harris started")
sar_harris_function_1,gradient_1,angle_1 = build_scale.build_scale(gray1,sigma,Mmax,ratio,d)
time_harris_function = time.time()
print("Create SAR HARRIS function Spend time:",time_harris_function-time1)

#Feaarure point detection
GR_key_array_1 = find_scale_extreme.find_scale_extreme(sar_harris_function_1,d_SH_1,sigma,ratio,gradient_1,angle_1)
time_point = time.time()

#calculating descriptors
descriptors_1 = calc_descriptors_copy.calc_descriptors(gradient_1,angle_1,GR_key_array_1)
time_descriptor = time.time()
print("calculating descriptor:", time_descriptor-time_point)

kp1,des1 = match_copy.delete_duplications(GR_key_array_1[:,0:2],descriptors_1)

print(kp1)
print(f"Liczba kp:\t{len(GR_key_array_1[:,0:2])}")
#plt.scatter(kp1[:,0],kp1[:,1])
#plt.imshow(img1,cmap="gray")
#plt.show()
binary_file_path = 'data_kp.npy'
np.save(binary_file_path, kp1)
binary_file_path = 'data_des.npy'
np.save(binary_file_path, des1)