# Biblioteki
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import RMSE


# Dane
title ="UDYE"
# DANE kontrolne
ptk_PNEO_k = np.genfromtxt(f"RefPoints/UTM_{title}_PNEO.csv", delimiter=',',dtype=np.float32)
print(ptk_PNEO_k)
ptk_CAPELLA_k = np.genfromtxt(f"RefPoints/UTM_{title}_CAPELLA.csv", delimiter=',',dtype=np.float32)
print(ptk_CAPELLA_k)
# Dane do wizualizacji
img1 = cv2.imread(f"Norm/SAR_{title}_SUB_035m_gray.png",0)
img2 = cv2.imread(f"Norm/EO_{title}_SUB_035m_gray.png",0)

# Dane do wyznaczenia referencyjej transformacji
ptk_PNEO_ref = np.genfromtxt(f"RefPoints/UTM_{title}_PNEO_ref.csv", delimiter=',',dtype=np.float32)
print(ptk_PNEO_ref)
ptk_CAPELLA_ref = np.genfromtxt(f"RefPoints/UTM_{title}_CAPELLA_ref.csv", delimiter=',',dtype=np.float32)
print(ptk_CAPELLA_ref)


diff = ptk_PNEO_k-ptk_CAPELLA_k
# RMSE i Transformacja
print(diff)
diff_ref = ptk_PNEO_ref-ptk_CAPELLA_ref
print(diff_ref)
# dist = np.linalg.norm(diff, axis=1)
# print(dist)
# print(np.mean(dist))
#Referencyjna Macierz transformacji
M,mask =cv2.estimateAffine2D(ptk_CAPELLA_ref,ptk_PNEO_ref)
#Błąd RMSE na punktach kontrolnych po transformacji affinicznej
rmse,blad = RMSE.calculate_RMSE(M,ptk_CAPELLA_k,ptk_PNEO_k)
#Błąd RMSE na punktach referencyjnych do transformacji affinicznej
rmse_ref,blad_ref = RMSE.calculate_RMSE(M,ptk_CAPELLA_ref,ptk_PNEO_ref)
print(blad*0.35)
print(rmse*0.35)


print(blad_ref*0.35)
print(rmse_ref*0.35)
# Figure
kolory = ["yellow","orange","red"]
# kolory_2 = ["green","cyan","blue"]
kolor = mc.LinearSegmentedColormap.from_list('mycmap', kolory)
# kolor_2 = mc.LinearSegmentedColormap.from_list('mycmap', kolory_2)
plt.title(title)
plt.scatter(ptk_PNEO_k[:,0],ptk_PNEO_k[:,1], c=blad*0.35,cmap=kolor)
plt.colorbar()
plt.scatter(ptk_PNEO_ref[:,0],ptk_PNEO_ref[:,1])
plt.imshow(img2,cmap="gray")
plt.show()

# #Wisualization of ERROR on Refpoints
# plt.scatter(ptk_PNEO_ref[:,0],ptk_PNEO_ref[:,1], c=blad_ref*0.35,cmap=kolor_2)
# plt.colorbar()

# plt.imshow(img2,cmap="gray")
# plt.show()



#TEST
h,w = img2.shape
image = cv2.warpAffine(img1,M,(h,w))
plt.imshow(image,cmap="gray")
plt.show()
#plt.imshow(img1,cmap="gray")
#plt.show()
