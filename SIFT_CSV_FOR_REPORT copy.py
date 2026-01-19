#   LIBRARIES
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import time
from Quality import RMSE
from Quality import Calc_and_Visual
import random
np.random.seed(0)
random.seed(0)
cv2.setRNGSeed(0)
data = ["URRC","UIAA","URWH","UDYE"]
scales = ["10","1","035","GM_035"]
norms = ["gray","log","bad"]
# if os.path.exists(f"report_SIFT/report_SIFT.html"):
#     print("Istnieje !!!")
# else:
#     with open(f"report_SIFT/report_SIFT.html", "w") as raport:
for d in data:
    print(f"Start {d}")
    for s,scale in enumerate(scales):
        for n,norm in enumerate(norms):
                # SEED
                np.random.seed(0)
                random.seed(0)
                cv2.setRNGSeed(0)
                total_time = 0
                # Wypisz Wymaluj
                print(f"SAR_{d}_SUB_{scale}m_{norm} -> EO_{d}_SUB_{scale}m_gray.png")
                # Dane
                img1 = cv2.imread(f"Norm/SAR_{d}_SUB_{scale}m_{norm}.png",0)
                if scale =="GM_035":
                    img2 = cv2.imread(f"Norm/EO_{d}_SUB_035m_gray.png",0)
                else:
                    img2 = cv2.imread(f"Norm/EO_{d}_SUB_{scale}m_gray.png",0)
                #SIFT Inicjalizacja
                start_time = time.time()
                sift = cv2.SIFT_create()
                end_time = time.time()
                sift_init_time = end_time - start_time 
                total_time+=sift_init_time
                print("Init SIFT time:\t",sift_init_time)
                #   START Detekcji i deskrypcji
                start_time = time.time()
                kp1, des1 = sift.detectAndCompute(img1, None)
                kp2, des2 = sift.detectAndCompute(img2, None)
                end_time = time.time()
                sift_detect_time = end_time - start_time 
                total_time+=sift_detect_time
                print("Detect SIFT time:\t",sift_detect_time)
                print("KP1:", len(kp1))
                print("KP2:", len(kp2))
                kppt1 = np.float32([k.pt for k in kp1]).reshape(-1, 2)
                kppt2 = np.float32([k.pt for k in kp2]).reshape(-1, 2)
                np.savetxt(f"report_SIFT/SAR_{d}_SUB_{scale}m_{norm}_before_mach_3.csv", kppt1, delimiter=",")
                np.savetxt(f"report_SIFT/EO_{d}_SUB_{scale}m_gray_before_mach_#.csv", kppt2, delimiter=",")
                #   MACHING
                #   FLANN
                start_time = time.time()
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                end_time = time.time()
                flann_init_time = end_time - start_time 
                total_time+=flann_init_time
                print("FLANN initial time:\t",flann_init_time)

                #   KNN MACHING
                start_time = time.time()
                matches = flann.knnMatch(des1, des2, k=2)
                end_time = time.time()
                flann_maching_time = end_time - start_time 
                total_time+=flann_maching_time
                print("FLANN Maching time:\t",flann_init_time)
                #   RATIO TEST LOWE
                start_time = time.time()
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                end_time = time.time()
                ratio_test_time = end_time - start_time 
                total_time+=ratio_test_time

                #   RANSAC
                start_time = time.time()
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                end_time = time.time()
                RANSAC_init_time = end_time - start_time 
                total_time+=RANSAC_init_time
                start_time = time.time()
                # Linczenie Macierzy Transformacji dla modelu po dopasowaniu obrazów
                if len(pts1) >=3:
                    M, mask = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                else:
                    print("Not enought points")
                    continue
                if M is None:
                    print("transformation impossible")
                    continue
                # Maska Po Ransach

                mask = mask.ravel().astype(bool)
                src_pts = pts1[mask]
                dst_pts = pts2[mask]
                end_time = time.time()
                RANSAC_model_time = end_time - start_time 
                total_time+=RANSAC_model_time
                np.savetxt(f"report_SIFT/SAR_{d}_SUB_{scale}m_{norm}_mach_3.csv", src_pts, delimiter=",")
                np.savetxt(f"report_SIFT/EO_{d}_SUB_{scale}m_{norm}_mach_3.csv", dst_pts, delimiter=",")
                total_time_LIST = [total_time, RANSAC_model_time,RANSAC_init_time, flann_maching_time, sift_init_time]
                np.savetxt(f"report_SIFT/SAR_{d}_SUB_{scale}m_{norm}_mach_time_3.csv", total_time_LIST, delimiter=",")
                # with open('report_SIFT/EO_SAR_SIFT_mach.csv','a') as report_time:
                #      report_time.write(str(total_time)+',')
                

print("DONE")