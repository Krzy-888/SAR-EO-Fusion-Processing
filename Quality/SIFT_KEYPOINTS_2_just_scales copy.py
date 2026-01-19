import matplotlib.pyplot as plt
import numpy as np
import cv2
sift = cv2.SIFT_create()
#draw Keypoints

paths = [r"Norm/CAPELLA_C06_UIAA_10_M_Subset_gray.png",r"Norm/CAPELLA_C06_UIAA_1_M_Subset_gray.png",r"Norm/CAPELLA_C06_UIAA_035_M_Subset_gray.png"]
parts = [100,1000,2857]
imgs = []
paths2 = [r"Norm/IMG_PNEO4_UIAA_10m_subset_3x3_gray.png",r"Norm/IMG_PNEO4_UIAA_1m_subset_3x3_gray.png",r"Norm/IMG_PNEO4_UIAA_035m_subset_3x3_gray.png"]
for i,path in enumerate(paths):
    img = cv2.imread(path,0)
    imgs.append(img[:parts[i],:parts[i]])
name = ["10m","1m","0,35m"]
fig, axes = plt.subplots(3, 2)
fig.suptitle("SIFT")
for i,im in enumerate(imgs):
    #title = name[i] + f" Znaleziono: {len(kp)}"
    im2 = cv2.imread(paths2[i],0)
    im2 = im2[:parts[i],:parts[i]]
    if i == 0:
        axes[i,0].set_title("SAR")
        axes[i,1].set_title("EO")
    axes[i,0].set_ylabel(name[i])
    axes[i,0].imshow(im,cmap="gray")
    axes[i,1].imshow(im2,cmap="gray")

plt.tight_layout()
plt.show()
