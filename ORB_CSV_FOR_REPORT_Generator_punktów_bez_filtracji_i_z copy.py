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
data = ["URRC"]
scales = ["035"]
norms = ["gray"]
# if os.path.exists(f"report_SIFT/report_SIFT.html"):
#     print("Istnieje !!!")
# else:
#     with open(f"report_SIFT/report_SIFT.html", "w") as raport:
przedział = [1000,2000]
for d in data:
    print(f"Start {d}")
    for s,scale in enumerate(scales):
        for n,norm in enumerate(norms):
                np.random.seed(0)
                random.seed(0)
                cv2.setRNGSeed(0)
                total_time = 0
                # Wypisz Wymaluj
                print(f"SAR_{d}_SUB_{scale}m_{norm} -> EO_{d}_SUB_{scale}m_gray.png")
                # Dane
                img1= cv2.imread(f"Norm/SAR_{d}_SUB_{scale}m_{norm}.png",0)
                img2 = cv2.imread(f"Norm/EO_{d}_SUB_{scale}m_gray.png",0)

                img1 = img1[przedział[0]:przedział[1],przedział[0]:przedział[1]]
                img2 = img2[przedział[0]:przedział[1],przedział[0]:przedział[1]]
                #SIFT Inicjalizacja
                orb = cv2.ORB_create(nfeatures = 800000)
                #   START Detekcji i deskrypcji
                kp1, des1 = orb.detectAndCompute(img1, None)
                kp2, des2 = orb.detectAndCompute(img2, None)
                print("KP1:", len(kp1))
                print("KP2:", len(kp2))
                kppt1 = np.float32([k.pt for k in kp1]).reshape(-1, 2)
                kppt2 = np.float32([k.pt for k in kp2]).reshape(-1, 2)
                # FLANN_INDEX_KDTREE = 1
                # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                # search_params = dict(checks=50)
                # flann = cv2.FlannBasedMatcher(index_params, search_params)
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm = FLANN_INDEX_LSH,
                table_number = 6,
                key_size = 12,
                multi_probe_level = 1)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                #   KNN MACHING
                matches = flann.knnMatch(des1, des2, k=2)

                good_matches = []
                # for m, n in matches:
                #     if m.distance < 0.7 * n.distance:
                #         good_matches.append(m)
                for mach in matches:
                    if len(mach) !=2:
                        continue
                    else:
                        m,n = mach
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                #   RANSAC
                pts1_t = np.float32([kp1[m.queryIdx].pt for m,n in matches]).reshape(-1, 2)
                pts2_t = np.float32([kp2[m.trainIdx].pt for m,n in matches]).reshape(-1, 2)
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                # Linczenie Macierzy Transformacji dla modelu po dopasowaniu obrazów
                if len(pts1) >=3:
                    M, mask = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    M, mask_2 = cv2.estimateAffine2D(pts1_t, pts2_t, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                mask = mask.ravel().astype(bool)
                mask_2 = mask_2.ravel().astype(bool)
                src_pts = pts1[mask]
                dst_pts = pts2[mask]
                src_pts_t = pts1_t[mask_2]
                dst_pts_t = pts2_t[mask_2]
                # points = [[pts1_t,pts2_t],[src_pts_t,dst_pts_t],[pts1,pts2],[src_pts,dst_pts]]
                np.savetxt('Filter_test/test_pts1.csv',pts1,delimiter=",")
                np.savetxt('Filter_test/test_pts2.csv',pts2,delimiter=",")
                np.savetxt('Filter_test/test_pts1_t.csv',pts1_t,delimiter=",")
                np.savetxt('Filter_test/test_pts2_t.csv',pts2_t,delimiter=",")
                np.savetxt('Filter_test/src_pts.csv',src_pts,delimiter=",")
                np.savetxt('Filter_test/dst_pts.csv',dst_pts,delimiter=",")
                np.savetxt('Filter_test/src_pts_t.csv',src_pts_t,delimiter=",")
                np.savetxt('Filter_test/dst_pts_t.csv',dst_pts_t,delimiter=",")
# fig, axes = plt.subplots(2, 4)
# for p,po in enumerate(points):
#     h,w = img1.shape
#     matrix,_ = cv2.estimateAffine2D(po[0],po[1])
#     print(f"{p} matrix ok")
#     image = cv2.warpAffine(img1,matrix,(h,w))
#     print(f"{p} wrapy ok")
#     Calc_and_Visual.show_maches_in_axis(axes[0][p],img1,img2,po[0],po[1],'m')
#     print(f"{p} calkuluj ok")
#     axes[1][p].imshow(image,cmap="gray")
#     print(f"{p} imshow ok")
# plt.tight_layout()
# plt.show()    



                

print("DONE")