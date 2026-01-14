#   LIBRARIES
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import time
from Quality import RMSE
from Quality import Calc_and_Visual
import random

import build_scale
import find_scale_extreme
import calc_descriptors
import match

import ransac





np.random.seed(0)
random.seed(0)
cv2.setRNGSeed(0)
data = ["URRC","UIAA","URWH","UDYE"]
scales_es = ["10","1"]
norms = ["gray","log","bad"]
# if os.path.exists(f"SAR-SIFT/report_SAR_SIFT/SAR-SIFT/report_SAR_SIFT.html"):
#     print("Istnieje !!!")
# else:
#     with open(f"SAR-SIFT/report_SAR_SIFT/SAR-SIFT/report_SAR_SIFT.html", "w") as raport:
#    scaleses bo scales zajęte, dATESES TEŻ
for dATESES in data:
    print(f"Start {dATESES}")
    for s,scaleses in enumerate(scales_es):
        for n,norm in enumerate(norms):
                # SEED
                np.random.seed(0)
                random.seed(0)
                cv2.setRNGSeed(0)
                total_time = 0
                # Wypisz Wymaluj
                print(f"SAR_{dATESES}_SUB_{scaleses}m_{norm} -> EO_{dATESES}_SUB_{scaleses}m_gray.png")
                # Dane
                gray1 = cv2.imread(f"Norm/SAR_{dATESES}_SUB_{scaleses}m_{norm}.png",0)
                if scaleses =="GM_035":
                    gray2 = cv2.imread(f"Norm/EO_{dATESES}_SUB_035m_gray.png",0)
                else:
                    gray2 = cv2.imread(f"Norm/EO_{dATESES}_SUB_{scaleses}m_gray.png",0)
                
                #IMAGES PRE PROCESING

                # gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
                # gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
                gray1 = gray1/255.
                gray2 = gray2/255.

                #SIFT Inicjalizacja


                start_time = time.time()
                sigma = 2                      #initial layer scale
                ratio = 2**(1/3.)               #scale ratio
                Mmax = 8                       #layer number
                d = 0.04
                d_SH_1 = 0.8                   #Harros function threshold
                d_SH_2 = 0.8                   #Harros function threshold
                distRatio = 0.9
                error_threshold = 1
                end_time = time.time()
                sift_init_time = end_time - start_time 
                total_time+=sift_init_time
                print("Init SIFT time:\t",sift_init_time)
                #   START Detekcji i deskrypcji
                start_time = time.time()
                sar_harris_function_1,gradient_1,angle_1 = build_scale.build_scale(gray1,sigma,Mmax,ratio,d)
                sar_harris_function_2,gradient_2,angle_2 = build_scale.build_scale(gray2,sigma,Mmax,ratio,d)
                GR_key_array_1 = find_scale_extreme.find_scale_extreme(sar_harris_function_1,d_SH_1,sigma,ratio,gradient_1,angle_1)
                GR_key_array_2 = find_scale_extreme.find_scale_extreme(sar_harris_function_2,d_SH_2,sigma,ratio,gradient_2,angle_2)
                descriptors_1, locs_1 = calc_descriptors.calc_descriptors(gradient_1,angle_1,GR_key_array_1)
                descriptors_2, locs_2 = calc_descriptors.calc_descriptors(gradient_2,angle_2,GR_key_array_2)
                end_time = time.time()
                sift_detect_time = end_time - start_time 
                total_time+=sift_detect_time
                print("Detect SIFT time:\t",sift_detect_time)
                print("KP1:", len(locs_1))
                print("KP2:", len(locs_2))
                np.savetxt(f"SAR-SIFT/report_SAR_SIFT/SAR_{dATESES}_SUB_{scaleses}m_{norm}_before_mach.csv", locs_1[:,0:2], delimiter=",")
                np.savetxt(f"SAR-SIFT/report_SAR_SIFT/EO_{dATESES}_SUB_{scaleses}m_gray_before_mach.csv", locs_2[:,0:2], delimiter=",")
                #   MACHING
                start_time = time.time()
                kp1,kp2,des1,des2 = match.delete_duplications(GR_key_array_1[:,0:2],GR_key_array_2[:,0:2],descriptors_1,descriptors_2)
                good_kp1,good_kp2 = match.deep_match(kp1,kp2,des1,des2,distRatio)
                if len(good_kp1) >=3:
                    better_kp1,better_kp2 = ransac.ransac(good_kp1,good_kp2,error_threshold)
                    solution1,rmse = ransac.least_square(better_kp1,better_kp2)
                else:
                    print("Not enought points")
                    continue
                RANSAC_init_time = end_time - start_time 
                total_time+=RANSAC_init_time
                start_time = time.time()
                # Linczenie Macierzy Transformacji dla modelu po dopasowaniu obrazów
                better_kp1 = np.asarray(better_kp1, dtype=np.float32)
                better_kp2 = np.asarray(better_kp2, dtype=np.float32)
                if len(good_kp1) >=3:
                    M, mask = cv2.estimateAffine2D(better_kp1[:,0:2], better_kp2[:,0:2], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                else:
                    print("Not enought points")
                    continue
                if M is None:
                    print("transformation impossible")
                    continue
                np.savetxt(f"SAR-SIFT/report_SAR_SIFT/SAR_{dATESES}_SUB_{scaleses}m_{norm}_mach.csv", better_kp1, delimiter=",")
                np.savetxt(f"SAR-SIFT/report_SAR_SIFT/EO_{dATESES}_SUB_{scaleses}m_{norm}_mach.csv", better_kp2, delimiter=",")
                with open('SAR-SIFT/report_SAR_SIFT/EO_SAR_ORB_mach.csv','a') as report_time:
                     report_time.write(str(total_time)+',')
print("DONE")