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
przedział = [0,2000]
for d in data:
    print(f"Start {d}")
    for s,scale in enumerate(scales):
        for n,norm in enumerate(norms):
                np.random.seed(0)
                random.seed(0)
                cv2.setRNGSeed(0)
                total_time = 0
                print(f"SAR_{d}_SUB_{scale}m_{norm} -> EO_{d}_SUB_{scale}m_gray.png")
                img1= cv2.imread(f"Norm/SAR_{d}_SUB_{scale}m_{norm}.png",0)
                img2 = cv2.imread(f"Norm/EO_{d}_SUB_{scale}m_gray.png",0)

                img1 = img1[przedział[0]:przedział[1],przedział[0]:przedział[1]]
                img2 = img2[przedział[0]:przedział[1],przedział[0]:przedział[1]]
nazwy = ["Bez filtracji", "Lowe ratio test", "RANSAC", "Lowe ratio test + RANSAC"]
pliki = [["Filter_test/test_pts1_t.csv","Filter_test/test_pts2_t.csv"],
         ["Filter_test/test_pts1.csv","Filter_test/test_pts2.csv"],
         ["Filter_test/src_pts_t.csv","Filter_test/dst_pts_t.csv"],
         ["Filter_test/src_pts.csv","Filter_test/dst_pts.csv"]]
fig, axes = plt.subplots(2, 4)
for p,po in enumerate(pliki):
    np.random.seed(0)
    random.seed(0)
    cv2.setRNGSeed(0)
    print('ok')
    h,w = img1.shape
    ptk_CAP = np.genfromtxt(po[0], delimiter=',',dtype=np.float32)
    ptk_Pneo = np.genfromtxt(po[1], delimiter=',',dtype=np.float32)

    matrix,_ = cv2.estimateAffine2D(ptk_CAP,ptk_Pneo)
    print(f"{p} matrix ok")
    image = cv2.warpAffine(img1,matrix,(h,w))
    print(f"{p} wrapy ok")
    axes[0][p].set_title(nazwy[p])
    Calc_and_Visual.show_maches_in_axis_2(axes[0][p],img1,img2,ptk_CAP,ptk_Pneo,'m')
    print(f"{p} calkuluj ok")
    axes[1][p].imshow(image,cmap="gray")
    print(f"{p} imshow ok")
plt.tight_layout()
plt.show()    



                

print("DONE")