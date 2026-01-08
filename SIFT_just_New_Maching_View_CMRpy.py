#   LIBRARIES
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import time
from Quality import RMSE
from Quality import Calc_and_Visual
total_time = 0
#   SIFT
start_time = time.time()
sift = cv2.SIFT_create()
end_time = time.time()
sift_init_time = end_time - start_time 
total_time+=sift_init_time
print("Initial SIFT time:\t",sift_init_time)
#   WCZYTANIE DANYCH
img1 = cv2.imread(r"Norm/SAR_URRC_SUB_1m_gray.png",0)
img2 = cv2.imread(r"Norm/EO_URRC_SUB_1m_gray.png",0)
ptk_PNEO = np.genfromtxt(r"RefPoints/UTM_URRC_PNEO.csv", delimiter=',',dtype=np.float32)
print(ptk_PNEO)
ptk_CAPELLA = np.genfromtxt(r"RefPoints/UTM_URRC_CAPELLA.csv", delimiter=',',dtype=np.float32)
print(ptk_CAPELLA)

print("\t***img1***")
print(img1.shape)
print(img1.max())
print(img1.min())
print("\t***img2***")
print(img2.shape)
print(img2.max())
print(img2.min())

img1 = img1[0:1500, 0:1500]
img2 = img2[0:1500, 0:1500]

#   START
start_time = time.time()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
end_time = time.time()
sift_detect_time = end_time - start_time 
total_time+=sift_detect_time
print("Detect SIFT time:\t",sift_detect_time)
print("KP1:", len(kp1))
print("KP2:", len(kp2))
out1 = cv2.drawKeypoints(img1, kp1, None)
out2 = cv2.drawKeypoints(img2, kp2, None)

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
print("Ratio test time:\t",ratio_test_time)
print("Liczba dobrych dopasowań:", len(good_matches))

#   RANSAC
start_time = time.time()
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
end_time = time.time()
RANSAC_init_time = end_time - start_time 
total_time+=RANSAC_init_time
print("RANSAC_init_time:\t",ratio_test_time)

start_time = time.time()
M, mask = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=2.0)

print(M)
mask = mask.ravel().astype(bool)
src_pts = pts1[mask]
dst_pts = pts2[mask]
M2 = cv2.getAffineTransform(src_pts[:3],dst_pts[:3])
print(M2)
end_time = time.time()
RANSAC_model_time = end_time - start_time 
total_time+=RANSAC_model_time
print("RANSAC_model_time:\t",ratio_test_time)

# mask = 1 czyli dobre dopasowanie
good_matches_masked = [m for m, inlier in zip(good_matches, mask) if inlier]
bad_matches_masked  = [m for m, inlier in zip(good_matches, mask) if not inlier]

N_corr = np.sum(mask)
N_maches = len(mask)
CMI = N_corr/ N_maches
mask = mask.ravel().astype(bool)
src_pts = pts1[mask]
dst_pts = pts2[mask]

#print(M)
rmse_1,blad = RMSE.calculate_RMSE(M,src_pts,dst_pts)
rmse_2,blad = RMSE.calculate_RMSE(M,ptk_CAPELLA,ptk_PNEO)

CMR_corr,rmse_3,blad_3 =  RMSE.calculate_CMR(ptk_CAPELLA,ptk_PNEO,src_pts,dst_pts,1)
#print("RMSE:\t", RMSE*0.35)
#PRZED
h,w = img1.shape
image = cv2.warpAffine(img1,M,(h,w))
#image = cv2.warpAffine(img1,M2,(h,w))
tytuły = ["złe","dobre","wynik transformacji"]
points = [pts1[~mask],pts1[mask],pts2[~mask],pts2[mask]]
color = ['r','g']
fig, axes = plt.subplots(3, 1)
for i in range(3):
    if i == 2:
        axes[i].imshow(image,cmap="gray")
    else:
        Calc_and_Visual.show_maches_in_axis(axes[i],img1,img2,points[i],points[i+2],color[i])
    axes[i].set_title(tytuły[i])
plt.tight_layout()
plt.show()


CMR_corr,rmse_3,blad_3,corr_mask =  RMSE.calculate_CMR_mask(ptk_CAPELLA,ptk_PNEO,src_pts,dst_pts,1)
#corr_mask = corr_mask.ravel().astype(bool)
#print("RMSE:\t", RMSE*0.35)
#PRZED
# h,w = img1.shape
# image = cv2.warpAffine(img1,M,(h,w))
#image = cv2.warpAffine(img1,M2,(h,w))
tytuły = ["złe","dobre","wynik transformacji"]
#points = [pts1[~corr_mask],pts1[corr_mask],pts2[~corr_mask],pts2[corr_mask]]

corr_mask = corr_mask.ravel().astype(bool)

bad_src  = src_pts[~corr_mask]
good_src = src_pts[corr_mask]

bad_dst  = dst_pts[~corr_mask]
good_dst = dst_pts[corr_mask]

points = [bad_src, good_src, bad_dst, good_dst]
color = ['r','g']
fig, axes = plt.subplots(3, 1)
for i in range(3) :
    if i == 2:
        axes[i].imshow(image,cmap="gray",origin="upper")
    else:
        Calc_and_Visual.show_maches_in_axis(axes[i],img1,img2,points[i],points[i+2],color[i])
    axes[i].set_title(tytuły[i])
plt.tight_layout()
plt.show()
#plt.show()


#for point in kp1:
#    print(point.pt)
"""plt.title("złe")
plt.imshow(result)
plt.subplot(3,1,2)
plt.title("dobre")
plt.imshow(result2)
plt.subplot(3,1,3)
plt.title("dobre")
plt.imshow(image,cmap="gray")
plt.show()"""
print(f"N corr:\t{N_corr}\nN maches:\t{N_maches}\nCMR: {CMI*100}\nCMR corr:\t{CMR_corr}\nRMSE: \t{rmse_1*0.35}\nRMSE Kontrol:\t{rmse_2*0.35}\nTotal Time: \t{total_time}")
print(f"CMR corr:\t{CMR_corr}\nRMSE: \t{rmse_3*0.35}")
print(blad_3)