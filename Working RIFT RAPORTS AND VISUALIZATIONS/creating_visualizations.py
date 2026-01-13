import RMSE
import Calc_and_Visual
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Metody PUNKTY DOPASOWANIA
ptk_CAP = np.genfromtxt(r"report_RIFT/SAR_UIAA_SUB_1m_gray_mach.csv", delimiter=',',dtype=np.float32)
ptk_PNEO = np.genfromtxt(r"report_RIFT/EO_UIAA_SUB_1m_gray_mach.csv", delimiter=',',dtype=np.float32)

# Punkty Referencyjnej transformacji
ptk_CAP_ref = np.genfromtxt(r"RefPoints/UTM_UIAA_CAPELLA.csv", delimiter=',',dtype=np.float32)
ptk_PNEO_ref = np.genfromtxt(r"RefPoints/UTM_UIAA_PNEO.csv", delimiter=',',dtype=np.float32)
# Punkty Kontrone
ptk_CAP_check = np.genfromtxt(r"RefPoints/UTM_UIAA_CAPELLA.csv", delimiter=',',dtype=np.float32)
ptk_PNEO_check = np.genfromtxt(r"RefPoints/UTM_UIAA_PNEO.csv", delimiter=',',dtype=np.float32)
# Wczzytaanie obrazu
img1 = cv2.imread(r"Norm/EO_UIAA_SUB_1m_gray.png",0) 
img2 = cv2.imread(r"Norm/SAR_UIAA_SUB_1m_gray.png",0) 

iloraz = 10/0.35

tytuły = ["złe","dobre","wynik transformacji"]
ptk_PNEO_ref = ptk_PNEO_ref/iloraz
ptk_CAPELLA_ref = ptk_PNEO_ref/iloraz

# Recherencyjna macierz
M_ref, mask = cv2.estimateAffine2D(ptk_CAP_ref, ptk_PNEO_ref)

# Macierz powstała w skutek dopasowania automatyczną metodą
M_nowa, mask = cv2.estimateAffine2D(ptk_CAP_ref, ptk_PNEO_ref)

#Właściwa maska i właściwe  CMR
CMR,treshold,blad,mask = RMSE.calculate_CMR_mask_new(M_ref,ptk_CAP_check,ptk_PNEO_check,ptk_CAP,ptk_PNEO)

rmse = RMSE.calculate_RMSE(M_nowa,ptk_CAP_check,ptk_PNEO_check)
print(rmse)
print(CMR)
h,w = img2.shape
image = cv2.warpAffine(img1,M_nowa,(h,w))

points = [ptk_CAP[~mask],ptk_CAP[mask],ptk_PNEO[~mask],ptk_PNEO[mask]]
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