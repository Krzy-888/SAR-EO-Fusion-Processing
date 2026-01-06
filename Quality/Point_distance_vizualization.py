# Biblioteki
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import RMSE

# DANE
ptk_PNEO = np.genfromtxt(r"Norm/UTM_URRC_PNEO.csv", delimiter=',',dtype=np.float32)
print(ptk_PNEO)
ptk_CAPELLA = np.genfromtxt(r"Norm/UTM_URRC_CAPELLA.csv", delimiter=',',dtype=np.float32)
print(ptk_CAPELLA)
img1 = cv2.imread(r"Norm/CAPELLA_C05_URRC_Subset1000_2_gray.png",0)
img2 = cv2.imread(r"Norm/IMG_PNEO4_URRC_Subset1000_gray.png",0)

# RMSE i Transformacja
M,mask =cv2.estimateAffine2D(ptk_CAPELLA,ptk_PNEO)
rmse,blad = RMSE.calculate_RMSE(M,ptk_CAPELLA,ptk_PNEO)
print(blad)
print(rmse)

# Figure
kolory = ["yellow","orange","red"]
kolor = mc.LinearSegmentedColormap.from_list('mycmap', kolory)
plt.scatter(ptk_PNEO[:,0],ptk_PNEO[:,1], c=blad,cmap=kolor)
plt.colorbar()
plt.imshow(img2,cmap="gray")
plt.show()