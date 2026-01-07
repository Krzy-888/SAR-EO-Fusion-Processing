# Biblioteki
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import RMSE

# DANE
ptk_PNEO = np.genfromtxt(r"RefPoints/UTM_URWH_PNEO.csv", delimiter=',',dtype=np.float32)
print(ptk_PNEO)
ptk_CAPELLA = np.genfromtxt(r"RefPoints/UTM_URWH_CAPELLA.csv", delimiter=',',dtype=np.float32)
print(ptk_CAPELLA)
img1 = cv2.imread(r"Norm/SAR_URWH_SUB_035m_gray.png",0)
img2 = cv2.imread(r"Norm/EO_URWH_SUB_035m_gray.png",0)
diff = ptk_PNEO-ptk_CAPELLA
# RMSE i Transformacja
print(diff)
dist = np.linalg.norm(diff, axis=1)
print(dist)
print(np.mean(dist))
M,mask =cv2.estimateAffine2D(ptk_CAPELLA,ptk_PNEO)
rmse,blad = RMSE.calculate_RMSE(M,ptk_CAPELLA,ptk_PNEO)
print(blad*0.35)
print(rmse*0.35)

# Figure
kolory = ["yellow","orange","red"]
kolor = mc.LinearSegmentedColormap.from_list('mycmap', kolory)
plt.scatter(ptk_PNEO[:,0],ptk_PNEO[:,1], c=blad*0.35,cmap=kolor)
plt.colorbar()
plt.imshow(img2,cmap="gray")
plt.show()
#TEST
h,w = img2.shape
image = cv2.warpAffine(img1,M,(h,w))
plt.imshow(image,cmap="gray")
plt.show()
#plt.imshow(img1,cmap="gray")
#plt.show()
