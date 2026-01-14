# Biblioteki
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import RMSE

# Dane
title ="UIAA"
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

kolory = ["yellow","orange","red"]
kolor = mc.LinearSegmentedColormap.from_list('mycmap', kolory)
vmin = 0
vmax = 10
blad_min = []
blad_max = []
pkt_PNEO_list = []
pkt_CAPELLA_list = []
new_blad_list = []
grd = [10,1,0.35]
ilorazy = [grd[0]/0.35,grd[1]/0.35,grd[2]/0.35]
rmse_list = []
matrix_list =[]
for i,path in enumerate(paths2):
    n_pkt_neo = ptk_PNEO/ilorazy[i]
    #print(n_pkt_neo)
    pkt_PNEO_list.append(n_pkt_neo)
    n_pkt_cap = ptk_CAPELLA/ilorazy[i]
    #print(n_pkt_cap)
    pkt_CAPELLA_list.append(n_pkt_cap)
    M,mask =cv2.estimateAffine2D(n_pkt_cap,n_pkt_neo)
    rmse,blad = RMSE.calculate_RMSE(M,n_pkt_cap,n_pkt_neo)
    #print(M)
    blad_2 = np.array(blad)*grd[i]
    blad_min.append(min(blad_2))
    blad_max.append(max(blad_2))
    new_blad_list.append(blad_2)
    rmse_list.append(rmse*grd[i])
    matrix_list.append(M)
vmin = min(blad_min)
vmax = max(blad_max)
fig, axes = plt.subplots(3, 1,constrained_layout=True)

#axes[0].set_title("SAR")
for i,path in enumerate(paths2):
    pkt = pkt_PNEO_list[i]
    img = cv2.imread(path,0)
    sc = axes[i].scatter(pkt[:,0],pkt[:,1], c=new_blad_list[i],cmap=kolor,vmin=vmin,vmax=vmax)
    axes[i].imshow(img,cmap="gray")
    
    print(rmse_list[i])
    print(new_blad_list[i])

fig.colorbar(sc, ax=axes, orientation="vertical")
#plt.tight_layout()
plt.show()

# Visual Test
transform_list = []
for i,path in enumerate(paths2):
    img_EO = cv2.imread(path,0)
    img_SAR = cv2.imread(path.replace("EO_","SAR_"),0)
    h,w = img_EO.shape
    image = cv2.warpAffine(img_SAR,matrix_list[i],(h,w))
    transform_list.append(image)


fig2, axes2 = plt.subplots(3, 1)
for i,img in enumerate(transform_list):
    axes2[i].imshow(img,cmap="gray")
plt.tight_layout()
plt.show()
